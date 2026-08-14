"""
ui/panels/connection_panel.py
==================================================================
Connection & Hardware Management dock.

    * IP + all four UR ports (RTDE / Dashboard / Script / RT)
    * Model selector (UR3..UR30, CB3 & e-Series)
    * Live status: transport, robot mode, safety, ping, voltage
    * Dashboard controls (power / brake release / play / pause)
    * Emergency-Stop button and Freedrive toggle
==================================================================
"""
from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLineEdit, QComboBox, QPushButton,
    QLabel, QGroupBox, QHBoxLayout, QSpinBox, QFormLayout,
)

from robot.ur_bridge import URBridge, ConnectionConfig, RobotState, Transport
from robot.ur_models import MODEL_NAMES


_MODE_COLORS = {
    "RUNNING": "#a3be8c", "IDLE": "#ebcb8b", "FREEDRIVE": "#88c0d0",
    "ERROR": "#bf616a", "DISCONNECTED": "#4c566a",
}
_SAFETY_COLORS = {
    "NORMAL": "#a3be8c", "REDUCED": "#ebcb8b",
    "PROTECTIVE_STOP": "#bf616a", "EMERGENCY_STOP": "#bf616a",
    "SAFEGUARD_STOP": "#d08770", "FAULT": "#bf616a", "VIOLATION": "#bf616a",
}


class ConnectionPanel(QWidget):
    model_changed = Signal(str)
    estop_pressed = Signal()

    def __init__(self, bridge: URBridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self._build()
        bridge.state_updated.connect(self._on_state)
        bridge.connected_changed.connect(self._on_connected)

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        cfg = self.bridge.config

        # --- connection settings ---
        box = QGroupBox("Controller")
        form = QFormLayout(box)
        self.ip_edit = QLineEdit(cfg.ip)
        form.addRow("IP address", self.ip_edit)

        self.model_box = QComboBox()
        self.model_box.addItems(MODEL_NAMES)
        self.model_box.setCurrentText(cfg.model_name)
        self.model_box.currentTextChanged.connect(self._model_changed)
        form.addRow("Robot model", self.model_box)

        ports = QHBoxLayout()
        self.rtde_port = self._port(cfg.rtde_port)
        self.dash_port = self._port(cfg.dashboard_port)
        self.script_port = self._port(cfg.script_port)
        self.rt_port = self._port(cfg.rt_port)
        for lbl, sb in (("RTDE", self.rtde_port), ("Dash", self.dash_port),
                        ("Script", self.script_port), ("RT", self.rt_port)):
            col = QVBoxLayout()
            small = QLabel(lbl); small.setStyleSheet("font-size:10px;color:#888")
            col.addWidget(small); col.addWidget(sb)
            ports.addLayout(col)
        form.addRow("Ports", self._wrap(ports))
        root.addWidget(box)

        # --- connect buttons ---
        btns = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.bridge.disconnect_robot)
        self.disconnect_btn.setEnabled(False)
        btns.addWidget(self.connect_btn)
        btns.addWidget(self.disconnect_btn)
        root.addLayout(btns)

        # --- status ---
        status = QGroupBox("Status")
        grid = QGridLayout(status)
        self.lbl_transport = self._status_value("—")
        self.lbl_mode = self._status_value("DISCONNECTED")
        self.lbl_safety = self._status_value("—")
        self.lbl_ping = self._status_value("— ms")
        self.lbl_volt = self._status_value("0.0 V")
        for r, (name, w) in enumerate((
            ("Transport", self.lbl_transport), ("Robot mode", self.lbl_mode),
            ("Safety", self.lbl_safety), ("Ping", self.lbl_ping),
            ("Voltage", self.lbl_volt))):
            cap = QLabel(name); cap.setStyleSheet("color:#888")
            grid.addWidget(cap, r, 0)
            grid.addWidget(w, r, 1)
        root.addWidget(status)

        # --- dashboard ---
        dash = QGroupBox("Dashboard")
        dl = QGridLayout(dash)
        commands = [
            ("Power On", "power_on"), ("Power Off", "power_off"),
            ("Brake Release", "brake_release"), ("Play", "play"),
            ("Pause", "pause"), ("Stop", "stop"),
            ("Unlock P-Stop", "unlock_protective_stop"),
            ("Close Popup", "close_safety_popup"),
        ]
        for i, (label, cmd) in enumerate(commands):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, c=cmd: self.bridge.dashboard(c))
            dl.addWidget(b, i // 2, i % 2)
        root.addWidget(dash)

        # --- safety controls ---
        safety = QHBoxLayout()
        self.freedrive_btn = QPushButton("Freedrive")
        self.freedrive_btn.setCheckable(True)
        self.freedrive_btn.toggled.connect(self.bridge.set_freedrive)
        self.freedrive_btn.setStyleSheet(
            "QPushButton:checked{background:#88c0d0;font-weight:bold}")
        self.estop_btn = QPushButton("EMERGENCY  STOP")
        self.estop_btn.setStyleSheet(
            "background:#bf616a;color:white;font-weight:bold;padding:10px")
        self.estop_btn.clicked.connect(self._estop)
        safety.addWidget(self.freedrive_btn)
        safety.addWidget(self.estop_btn, 2)
        root.addLayout(safety)
        root.addStretch(1)

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _port(value: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(1, 65535)
        sb.setValue(value)
        sb.setMaximumWidth(70)
        return sb

    @staticmethod
    def _status_value(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-family:Consolas,monospace;font-weight:bold")
        return lbl

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget(); w.setLayout(layout); return w

    # ---- actions ----------------------------------------------------------
    def _current_config(self) -> ConnectionConfig:
        return ConnectionConfig(
            ip=self.ip_edit.text().strip(),
            rtde_port=self.rtde_port.value(),
            dashboard_port=self.dash_port.value(),
            script_port=self.script_port.value(),
            rt_port=self.rt_port.value(),
            model_name=self.model_box.currentText(),
        )

    def _connect(self) -> None:
        self.bridge.set_config(self._current_config())
        self.bridge.connect_robot()

    def _model_changed(self, name: str) -> None:
        self.bridge.set_model(name)
        self.model_changed.emit(name)

    def _estop(self) -> None:
        self.bridge.emergency_stop()
        self.estop_pressed.emit()

    # ---- state feedback ---------------------------------------------------
    def _on_connected(self, connected: bool) -> None:
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)

    def _on_state(self, st: RobotState) -> None:
        self.lbl_transport.setText(st.transport.value)
        self.lbl_mode.setText(st.mode.value)
        self.lbl_mode.setStyleSheet(
            f"font-family:Consolas;font-weight:bold;"
            f"color:{_MODE_COLORS.get(st.mode.value, '#d8dee9')}")
        self.lbl_safety.setText(st.safety.value)
        self.lbl_safety.setStyleSheet(
            f"font-family:Consolas;font-weight:bold;"
            f"color:{_SAFETY_COLORS.get(st.safety.value, '#a3be8c')}")
        ping = "— ms" if st.ping_ms != st.ping_ms else f"{st.ping_ms:.1f} ms"  # nan check
        self.lbl_ping.setText(ping)
        self.lbl_volt.setText(f"{st.robot_voltage:.1f} V")
        self.freedrive_btn.blockSignals(True)
        self.freedrive_btn.setChecked(st.is_freedrive)
        self.freedrive_btn.blockSignals(False)
