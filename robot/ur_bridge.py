"""
robot/ur_bridge.py
==================================================================
Threaded interface between the GUI and a Universal Robot controller.

Design
------
Qt's event loop must never block, so *all* network I/O lives on a
background ``QThread``:

    GUI thread                         Worker thread (URWorker)
    ----------                         ------------------------
    URBridge (QObject facade)  ──cmd──►  queued slot invocations
        ▲                                     │
        └──────── Qt signals ◄────────────────┘  (state @ poll_hz)

The GUI only ever calls :class:`URBridge` methods and connects to its
signals. Everything crossing the thread boundary does so through Qt's
queued connections, which are thread-safe — worker code never touches a
widget, and GUI code never touches a socket.

Backends
--------
Two transport layers, chosen automatically:

* **RTDE** (preferred) via the ``ur_rtde`` package — gives calibrated
  joint/TCP feedback at up to 500 Hz plus a clean control API and the
  dashboard client.
* **Raw sockets** (fallback) — the secondary/realtime interfaces on
  ports 30002/30003 for URScript execution, and 29999 for the dashboard.
  Feedback in this mode is limited (URScript is written, not read), so
  the digital twin runs from commanded/simulated state.

If neither the robot nor ``ur_rtde`` is present, the bridge stays in
``SIMULATED`` mode: commands update an internal virtual joint vector so
the whole UI — jogging, the 3D twin, program playback — remains usable
offline.
==================================================================
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import (
    QObject, QThread, QTimer, Qt, Signal, Slot, QMutex, QMutexLocker,
)

from .ur_models import URModel, get_model, DEFAULT_MODEL

# --- optional heavy dependency: imported lazily so the app still starts -----
try:  # pragma: no cover - availability depends on the host
    import rtde_control        # type: ignore
    import rtde_receive        # type: ignore
    import dashboard_client    # type: ignore
    _HAVE_RTDE = True
except Exception:              # noqa: BLE001 - any import failure => fallback
    _HAVE_RTDE = False


# ===========================================================================
#  Enums & value objects
# ===========================================================================
class Transport(str, Enum):
    RTDE = "RTDE"
    SOCKET = "Socket"
    SIMULATED = "Simulated"


class RobotMode(str, Enum):
    """Mirrors UR's robotmode register, plus local states."""
    DISCONNECTED = "DISCONNECTED"
    CONFIRM_SAFETY = "CONFIRM_SAFETY"
    BOOTING = "BOOTING"
    POWER_OFF = "POWER_OFF"
    POWER_ON = "POWER_ON"
    IDLE = "IDLE"
    BACKDRIVE = "BACKDRIVE"
    RUNNING = "RUNNING"
    FREEDRIVE = "FREEDRIVE"
    ERROR = "ERROR"


class SafetyStatus(str, Enum):
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"
    RECOVERY = "RECOVERY"
    SAFEGUARD_STOP = "SAFEGUARD_STOP"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    FAULT = "FAULT"
    VIOLATION = "VIOLATION"


@dataclass
class RobotState:
    """Snapshot of robot feedback, emitted to the GUI each poll cycle."""
    connected: bool = False
    transport: Transport = Transport.SIMULATED
    mode: RobotMode = RobotMode.DISCONNECTED
    safety: SafetyStatus = SafetyStatus.NORMAL
    # Feedback vectors
    q_actual: List[float] = field(default_factory=lambda: [0.0] * 6)      # rad
    qd_actual: List[float] = field(default_factory=lambda: [0.0] * 6)     # rad/s
    tcp_pose: List[float] = field(default_factory=lambda: [0.0] * 6)      # m + axis-angle rad
    tcp_force: List[float] = field(default_factory=lambda: [0.0] * 6)     # N / Nm
    digital_inputs: int = 0
    digital_outputs: int = 0
    ping_ms: float = float("nan")
    robot_voltage: float = 0.0
    is_estopped: bool = False
    is_freedrive: bool = False

    def copy(self) -> "RobotState":
        import copy
        return copy.deepcopy(self)


@dataclass
class ConnectionConfig:
    ip: str = "192.168.1.100"
    rtde_port: int = 30004
    dashboard_port: int = 29999
    script_port: int = 30002          # secondary interface
    rt_port: int = 30003              # realtime interface
    model_name: str = DEFAULT_MODEL
    poll_hz: float = 125.0            # feedback poll rate
    frequency: float = 125.0          # RTDE receive frequency


# A neutral, safe "ready" pose (radians) used for the virtual robot.
HOME_Q = [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]


# ===========================================================================
#  Worker — lives on the background thread, owns all sockets
# ===========================================================================
class URWorker(QObject):
    """
    Runs the polling loop and executes commands. Never touches the GUI.

    All public @Slot methods are invoked via queued connections from
    :class:`URBridge`, so they execute on *this* (worker) thread even
    though the caller is on the GUI thread.
    """

    state_updated = Signal(object)        # RobotState
    log = Signal(str, str)                # (level, message)  level in DEBUG/INFO/WARN/ERROR
    connected_changed = Signal(bool)
    finished = Signal()

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__()
        self._cfg = config
        self._model: URModel = get_model(config.model_name)
        self._running = False
        self._want_connect = False

        # backend handles
        self._rtde_c = None            # rtde_control.RTDEControlInterface
        self._rtde_r = None            # rtde_receive.RTDEReceiveInterface
        self._dash = None              # dashboard_client.DashboardClient
        self._script_sock: Optional[socket.socket] = None

        # virtual state for SIMULATED / SOCKET feedback
        self._sim_q = list(HOME_Q)
        self._sim_target_q = list(HOME_Q)

        self._state = RobotState()
        self._lock = QMutex()          # guards _sim_q / config swaps

    # ---- lifecycle --------------------------------------------------------
    @Slot()
    def start(self) -> None:
        """
        Arm the periodic poll on the worker thread's **event loop**.

        Critically this uses a ``QTimer`` rather than a blocking
        ``while`` loop. A blocking loop never returns control to the
        thread's event loop, so queued command slots (jog, moveJ/L,
        freedrive, dashboard, e-stop — all delivered via
        ``QMetaObject.invokeMethod``) would never be dispatched and every
        button in the UI would appear dead. With a timer, the event loop
        stays free to interleave commands between polls.
        """
        self._running = True
        self._period = 1.0 / max(self._cfg.poll_hz, 1.0)
        self._last_ping = 0.0
        self._timer = QTimer()
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._poll_once)
        self._timer.start(int(self._period * 1000))
        self.log.emit("INFO", "Worker started (event-loop poll @ "
                              f"{self._cfg.poll_hz:g} Hz).")

    @Slot()
    def _poll_once(self) -> None:
        """One poll cycle — runs on the worker thread between command slots."""
        if not self._running:
            return
        try:
            if self._want_connect and not self._state.connected:
                self._do_connect()
            if self._state.connected:
                self._poll_feedback()
            else:
                self._tick_simulation(self._period)
            self.state_updated.emit(self._state.copy())
        except Exception as exc:                           # noqa: BLE001
            self.log.emit("ERROR", f"Poll loop error: {exc!r}")
            self._state.mode = RobotMode.ERROR

    @Slot()
    def stop(self) -> None:
        self._running = False
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
        self._teardown()
        self.finished.emit()

    # ---- connection -------------------------------------------------------
    @Slot()
    def request_connect(self) -> None:
        self._want_connect = True

    @Slot()
    def request_disconnect(self) -> None:
        self._want_connect = False
        self._teardown()
        self._state.connected = False
        self._state.transport = Transport.SIMULATED
        self._state.mode = RobotMode.DISCONNECTED
        self.connected_changed.emit(False)
        self.log.emit("INFO", "Disconnected.")

    def _do_connect(self) -> None:
        ip = self._cfg.ip
        if _HAVE_RTDE:
            try:
                self._rtde_r = rtde_receive.RTDEReceiveInterface(ip, self._cfg.frequency)
                self._rtde_c = rtde_control.RTDEControlInterface(ip, self._cfg.frequency)
                self._dash = dashboard_client.DashboardClient(ip)
                try:
                    self._dash.connect()
                except Exception:                          # noqa: BLE001
                    self._dash = None                      # dashboard optional
                self._state.transport = Transport.RTDE
                self._state.connected = True
                self.connected_changed.emit(True)
                self.log.emit("INFO", f"Connected to {ip} via RTDE.")
                return
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("WARN", f"RTDE connect failed ({exc}); trying sockets.")
                self._teardown()

        # --- raw socket fallback (URScript push) ---------------------------
        try:
            self._script_sock = socket.create_connection(
                (ip, self._cfg.script_port), timeout=2.0
            )
            self._state.transport = Transport.SOCKET
            self._state.connected = True
            self.connected_changed.emit(True)
            self.log.emit(
                "INFO",
                f"Connected to {ip}:{self._cfg.script_port} (URScript socket). "
                "Feedback is simulated in this mode.",
            )
        except Exception as exc:                           # noqa: BLE001
            self._want_connect = False
            self._state.connected = False
            self._state.transport = Transport.SIMULATED
            self._state.mode = RobotMode.DISCONNECTED
            self.connected_changed.emit(False)
            self.log.emit("ERROR", f"Could not reach robot at {ip}: {exc}")

    def _teardown(self) -> None:
        for closer in (
            lambda: self._rtde_c and self._rtde_c.disconnect(),
            lambda: self._rtde_r and self._rtde_r.disconnect(),
            lambda: self._dash and self._dash.disconnect(),
            lambda: self._script_sock and self._script_sock.close(),
        ):
            try:
                closer()
            except Exception:                              # noqa: BLE001
                pass
        self._rtde_c = self._rtde_r = self._dash = None
        self._script_sock = None

    # ---- feedback ---------------------------------------------------------
    def _poll_feedback(self) -> None:
        st = self._state
        if st.transport is Transport.RTDE and self._rtde_r is not None:
            r = self._rtde_r
            st.q_actual = list(r.getActualQ())
            st.qd_actual = list(r.getActualQd())
            st.tcp_pose = list(r.getActualTCPPose())
            try:
                st.tcp_force = list(r.getActualTCPForce())
            except Exception:                              # noqa: BLE001
                pass
            st.digital_inputs = int(r.getActualDigitalInputBits())
            st.digital_outputs = int(r.getActualDigitalOutputBits())
            st.robot_voltage = float(getattr(r, "getActualRobotVoltage", lambda: 0.0)())
            self._map_rtde_modes(r)
            with QMutexLocker(self._lock):
                self._sim_q = list(st.q_actual)            # keep sim in sync
        else:
            # SOCKET mode: no read channel, run the virtual model.
            self._tick_simulation(1.0 / max(self._cfg.poll_hz, 1.0))

        # Ping is a blocking TCP connect — throttle it to ~1 Hz so it never
        # stalls the poll/command dispatch on the worker thread.
        now = time.perf_counter()
        if now - getattr(self, "_last_ping", 0.0) > 1.0:
            st.ping_ms = self._ping()
            self._last_ping = now

    def _map_rtde_modes(self, r) -> None:
        """Translate RTDE safety/robot mode integers to our enums."""
        st = self._state
        try:
            rm = int(r.getRobotMode())
            # UR robotmode: 5=IDLE, 7=RUNNING (per RTDE spec), 3=POWER_OFF ...
            st.mode = {
                -1: RobotMode.DISCONNECTED, 0: RobotMode.CONFIRM_SAFETY,
                1: RobotMode.BOOTING, 2: RobotMode.POWER_OFF,
                3: RobotMode.POWER_ON, 4: RobotMode.IDLE,
                5: RobotMode.IDLE, 6: RobotMode.BACKDRIVE,
                7: RobotMode.RUNNING,
            }.get(rm, RobotMode.IDLE)
        except Exception:                                  # noqa: BLE001
            pass
        try:
            sm = int(r.getSafetyMode())
            st.safety = {
                1: SafetyStatus.NORMAL, 2: SafetyStatus.REDUCED,
                3: SafetyStatus.PROTECTIVE_STOP, 4: SafetyStatus.RECOVERY,
                5: SafetyStatus.SAFEGUARD_STOP, 6: SafetyStatus.EMERGENCY_STOP,
                7: SafetyStatus.EMERGENCY_STOP, 8: SafetyStatus.FAULT,
                9: SafetyStatus.VIOLATION,
            }.get(sm, SafetyStatus.NORMAL)
            st.is_estopped = st.safety in (
                SafetyStatus.EMERGENCY_STOP, SafetyStatus.PROTECTIVE_STOP
            )
        except Exception:                                  # noqa: BLE001
            pass

    def _ping(self) -> float:
        """Lightweight liveness check (TCP connect RTT to the RTDE port)."""
        try:
            t0 = time.perf_counter()
            with socket.create_connection((self._cfg.ip, self._cfg.dashboard_port), 0.2):
                return (time.perf_counter() - t0) * 1000.0
        except Exception:                                  # noqa: BLE001
            return float("nan")

    # ---- virtual robot (offline / socket feedback) ------------------------
    def _tick_simulation(self, dt: float) -> None:
        """Advance the virtual joint vector toward its target (rate-limited)."""
        st = self._state
        with QMutexLocker(self._lock):
            q = np.array(self._sim_q)
            tgt = np.array(self._sim_target_q)
            qd_max = np.array(self._model.qd_max)
            step = qd_max * dt
            delta = np.clip(tgt - q, -step, step)
            q = q + delta
            self._sim_q = q.tolist()
            st.qd_actual = (delta / dt).tolist()
        st.q_actual = list(self._sim_q)
        if not st.connected:
            st.mode = RobotMode.FREEDRIVE if st.is_freedrive else RobotMode.IDLE
        # TCP pose is filled in by the bridge's kinematics; left as-is here.

    # ---- commands (executed on worker thread) -----------------------------
    def _send_script(self, script: str) -> None:
        """Push a URScript program/line to the robot over the active transport."""
        if self._rtde_c is not None:
            # ur_rtde exposes many primitives; for arbitrary scripts we can
            # still use the secondary interface if opened. Prefer sockets here.
            pass
        if self._script_sock is not None:
            payload = (script.rstrip("\n") + "\n").encode("utf-8")
            try:
                self._script_sock.sendall(payload)
                self.log.emit("DEBUG", f"URScript >> {script.splitlines()[0][:60]}")
                return
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("ERROR", f"Script send failed: {exc}")
        self.log.emit("DEBUG", f"[sim] would run URScript:\n{script}")

    @Slot(list, float, float)
    def move_j(self, q: list, speed: float, accel: float) -> None:
        q = self._clamp_joints(q)
        if self._rtde_c is not None:
            try:
                self._rtde_c.moveJ(q, speed, accel)
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("ERROR", f"moveJ failed: {exc}")
        else:
            self._send_script(
                f"movej([{','.join(f'{v:.6f}' for v in q)}], "
                f"a={accel:.3f}, v={speed:.3f})"
            )
        with QMutexLocker(self._lock):
            self._sim_target_q = list(q)

    @Slot(list, float, float)
    def move_l(self, pose: list, speed: float, accel: float) -> None:
        if self._rtde_c is not None:
            try:
                self._rtde_c.moveL(pose, speed, accel)
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("ERROR", f"moveL failed: {exc}")
        else:
            self._send_script(
                f"movel(p[{','.join(f'{v:.6f}' for v in pose)}], "
                f"a={accel:.3f}, v={speed:.3f})"
            )

    @Slot(list)
    def servo_target(self, q: list) -> None:
        """Set a jog target the virtual/real robot moves toward smoothly."""
        q = self._clamp_joints(q)
        with QMutexLocker(self._lock):
            self._sim_target_q = list(q)
        if self._rtde_c is not None:
            try:
                self._rtde_c.moveJ(q, 0.5, 1.0, asynchronous=True)
            except Exception:                              # noqa: BLE001
                pass

    @Slot(bool)
    def set_freedrive(self, enable: bool) -> None:
        self._state.is_freedrive = enable
        if self._rtde_c is not None:
            try:
                self._rtde_c.teachMode() if enable else self._rtde_c.endTeachMode()
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("ERROR", f"Freedrive toggle failed: {exc}")
        else:
            self._send_script("freedrive_mode()" if enable else "end_freedrive_mode()")
        self.log.emit("INFO", f"Freedrive {'ON' if enable else 'OFF'}.")

    @Slot()
    def emergency_stop(self) -> None:
        """Immediate stop — protective stop via control API or dashboard."""
        self._state.is_estopped = True
        self._state.safety = SafetyStatus.EMERGENCY_STOP
        self._state.mode = RobotMode.ERROR
        if self._rtde_c is not None:
            try:
                self._rtde_c.stopScript()
                self._rtde_c.stopJ(2.0)
            except Exception:                              # noqa: BLE001
                pass
        if self._dash is not None:
            try:
                self._dash.stop()
            except Exception:                              # noqa: BLE001
                pass
        if self._script_sock is not None:
            self._send_script("stopj(2.0)\nhalt")
        # freeze virtual target
        with QMutexLocker(self._lock):
            self._sim_target_q = list(self._sim_q)
        self.log.emit("ERROR", "EMERGENCY STOP issued.")

    @Slot(int, bool)
    def set_digital_output(self, pin: int, value: bool) -> None:
        if self._rtde_c is not None:
            try:
                self._rtde_c.setStandardDigitalOut(pin, value)
                return
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("ERROR", f"setDO failed: {exc}")
        self._send_script(f"set_standard_digital_out({pin}, {'True' if value else 'False'})")

    @Slot(str)
    def run_script(self, script: str) -> None:
        """Execute an arbitrary URScript program (from the code editor)."""
        self._send_script(script)

    @Slot(str)
    def dashboard(self, command: str) -> None:
        """Send a dashboard command (power on, brake release, play, ...)."""
        if self._dash is not None:
            try:
                fn = {
                    "power_on": self._dash.powerOn,
                    "power_off": self._dash.powerOff,
                    "brake_release": self._dash.brakeRelease,
                    "play": self._dash.play,
                    "pause": self._dash.pause,
                    "stop": self._dash.stop,
                    "unlock_protective_stop": self._dash.unlockProtectiveStop,
                    "close_safety_popup": self._dash.closeSafetyPopup,
                }.get(command)
                if fn:
                    fn()
                    self.log.emit("INFO", f"Dashboard: {command}")
                    return
            except Exception as exc:                       # noqa: BLE001
                self.log.emit("ERROR", f"Dashboard '{command}' failed: {exc}")
        else:
            self._raw_dashboard(command)

    def _raw_dashboard(self, command: str) -> None:
        """Dashboard over a plain socket when ur_rtde isn't available."""
        text = {
            "power_on": "power on", "power_off": "power off",
            "brake_release": "brake release", "play": "play",
            "pause": "pause", "stop": "stop",
            "unlock_protective_stop": "unlock protective stop",
            "close_safety_popup": "close safety popup",
        }.get(command, command)
        try:
            with socket.create_connection((self._cfg.ip, self._cfg.dashboard_port), 1.0) as s:
                s.recv(4096)                               # greeting
                s.sendall((text + "\n").encode())
                reply = s.recv(4096).decode(errors="replace").strip()
                self.log.emit("INFO", f"Dashboard '{text}' -> {reply}")
        except Exception as exc:                           # noqa: BLE001
            self.log.emit("DEBUG", f"[sim] dashboard '{text}' ({exc})")

    # ---- helpers ----------------------------------------------------------
    def _clamp_joints(self, q: Sequence[float]) -> List[float]:
        lo, hi = self._model.q_min, self._model.q_max
        return [float(np.clip(v, lo[i], hi[i])) for i, v in enumerate(q)]

    @Slot(str)
    def set_model(self, name: str) -> None:
        self._model = get_model(name)
        self.log.emit("INFO", f"Active model: {name}")

    @Slot(object)
    def set_tcp_pose(self, pose: list) -> None:
        """Called by the bridge to inject FK-computed TCP pose while simulating."""
        self._state.tcp_pose = list(pose)


# ===========================================================================
#  Bridge — the GUI-facing facade (lives on the GUI thread)
# ===========================================================================
class URBridge(QObject):
    """
    Public API used by the UI. Owns the worker + its QThread and re-emits
    worker signals so widgets connect to a single stable object.

    Typical use::

        bridge = URBridge(ConnectionConfig(ip="192.168.1.10"))
        bridge.state_updated.connect(self.on_state)
        bridge.start()
        bridge.connect_robot()
        ...
        bridge.move_j(target_q, speed=1.0, accel=1.4)
    """

    # re-emitted worker signals
    state_updated = Signal(object)          # RobotState
    log = Signal(str, str)
    connected_changed = Signal(bool)

    def __init__(self, config: Optional[ConnectionConfig] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._cfg = config or ConnectionConfig()
        self._thread = QThread()
        self._thread.setObjectName("URWorkerThread")
        self._worker = URWorker(self._cfg)
        self._worker.moveToThread(self._thread)

        # wire worker -> bridge (queued across threads automatically)
        self._worker.state_updated.connect(self.state_updated)
        self._worker.log.connect(self.log)
        self._worker.connected_changed.connect(self.connected_changed)
        self._thread.started.connect(self._worker.start)
        self._worker.finished.connect(self._thread.quit)

        self._latest = RobotState()
        self.state_updated.connect(self._cache_state)

    # ---- thread lifecycle -------------------------------------------------
    def start(self) -> None:
        if not self._thread.isRunning():
            self._thread.start()

    def shutdown(self) -> None:
        """Stop the worker and join the thread cleanly (call on app exit)."""
        try:
            # Flip the flag directly (a plain bool read is safe cross-thread)
            # so polling halts immediately, then ask the worker to tear down
            # its timer/sockets on its own thread before we join.
            self._worker._running = False
            _invoke(self._worker, "stop")
            self._thread.quit()
            self._thread.wait(3000)
        except Exception:                                  # noqa: BLE001
            pass

    # ---- config -----------------------------------------------------------
    @property
    def config(self) -> ConnectionConfig:
        return self._cfg

    @property
    def state(self) -> RobotState:
        return self._latest

    def set_config(self, config: ConnectionConfig) -> None:
        self._cfg = config
        self._worker._cfg = config      # worker reads config fields directly
        self.set_model(config.model_name)

    def _cache_state(self, st: RobotState) -> None:
        self._latest = st

    # ---- command forwarders (thread-safe via queued invocation) -----------
    # Using Qt signals to marshal calls onto the worker thread keeps every
    # socket write off the GUI thread without manual QMetaObject plumbing.
    connect_robot_sig = Signal()
    disconnect_robot_sig = Signal()

    def connect_robot(self) -> None:
        _invoke(self._worker, "request_connect")

    def disconnect_robot(self) -> None:
        _invoke(self._worker, "request_disconnect")

    def move_j(self, q: Sequence[float], speed: float = 1.05, accel: float = 1.4) -> None:
        _invoke(self._worker, "move_j", list(map(float, q)), float(speed), float(accel))

    def move_l(self, pose: Sequence[float], speed: float = 0.25, accel: float = 1.2) -> None:
        _invoke(self._worker, "move_l", list(map(float, pose)), float(speed), float(accel))

    def servo_target(self, q: Sequence[float]) -> None:
        _invoke(self._worker, "servo_target", list(map(float, q)))

    def set_freedrive(self, enable: bool) -> None:
        _invoke(self._worker, "set_freedrive", bool(enable))

    def emergency_stop(self) -> None:
        _invoke(self._worker, "emergency_stop")

    def set_digital_output(self, pin: int, value: bool) -> None:
        _invoke(self._worker, "set_digital_output", int(pin), bool(value))

    def run_script(self, script: str) -> None:
        _invoke(self._worker, "run_script", str(script))

    def dashboard(self, command: str) -> None:
        _invoke(self._worker, "dashboard", str(command))

    def set_model(self, name: str) -> None:
        _invoke(self._worker, "set_model", str(name))

    def set_tcp_pose(self, pose: Sequence[float]) -> None:
        _invoke(self._worker, "set_tcp_pose", list(map(float, pose)))

    @staticmethod
    def rtde_available() -> bool:
        return _HAVE_RTDE


# ---------------------------------------------------------------------------
# Helper: invoke a worker slot on its own thread via a queued connection.
# ---------------------------------------------------------------------------
from PySide6.QtCore import QMetaObject, Qt, Q_ARG  # noqa: E402


def _invoke(obj: QObject, method: str, *args) -> None:
    """
    Queue ``obj.method(*args)`` to run on ``obj``'s thread.

    Falls back to a direct call if the object has no thread affinity yet
    (e.g. during unit tests without an event loop).
    """
    q_args = [Q_ARG(_qt_type(a), a) for a in args]
    ok = QMetaObject.invokeMethod(obj, method, Qt.ConnectionType.QueuedConnection, *q_args)
    if not ok:  # pragma: no cover
        getattr(obj, method)(*args)


def _qt_type(value) -> str:
    """Map a Python value to the Qt meta-type string used by Q_ARG."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "QString"
    if isinstance(value, list):
        return "QVariantList"
    return "QVariant"
