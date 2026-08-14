"""
ui/panels/jog_panel.py
==================================================================
Teach-pendant style jogging dock.

    * Joint-space jog: 6 joints, +/- nudge buttons and live sliders,
      shown in degrees (toggle to radians).
    * Cartesian jog: X/Y/Z (mm) and Rx/Ry/Rz (deg), relative to the
      Base or Tool frame, solved to joints through the IK.
    * Teach-waypoint buttons that emit the current configuration/pose
      for the Program Builder to capture (MoveJ / MoveL / MoveP / Process).

The panel never blocks: jog commands go to ``bridge.servo_target`` /
``bridge.move_j`` which marshal onto the worker thread. Live joint/pose
read-outs come from the ``state_updated`` stream.
==================================================================
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QDoubleSpinBox, QComboBox, QCheckBox,
)

from robot.ur_bridge import URBridge, RobotState
from robot.kinematics import Kinematics, pose_to_matrix, matrix_to_pose

_JOINTS = ("Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3")
_CART = ("X", "Y", "Z", "Rx", "Ry", "Rz")


class JogPanel(QWidget):
    teach_waypoint = Signal(str)     # emits StepType value: MoveJ/MoveL/MoveP/ProcessPoint

    def __init__(self, bridge: URBridge, kin: Kinematics, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.kin = kin
        # Two distinct joint vectors, deliberately decoupled:
        #   _cmd_q    : the jog *target* the user is building up with the
        #               buttons/sliders. Never overwritten by feedback while
        #               the user is jogging, so nudges accumulate smoothly.
        #   _actual_q : the latest measured/simulated joint state, shown in
        #               the read-outs. Re-syncs _cmd_q on connect, freedrive,
        #               or a large external divergence (e.g. hand-guiding).
        self._cmd_q = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        self._actual_q = self._cmd_q.copy()
        self._synced = False
        self._use_deg = True
        self._build()
        bridge.state_updated.connect(self._on_state)

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        # global jog settings
        top = QHBoxLayout()
        self.deg_chk = QCheckBox("degrees")
        self.deg_chk.setChecked(True)
        self.deg_chk.toggled.connect(self._toggle_units)
        top.addWidget(self.deg_chk)
        top.addWidget(QLabel("Joint step"))
        self.jstep = QDoubleSpinBox()
        self.jstep.setRange(0.1, 45.0); self.jstep.setValue(5.0); self.jstep.setSuffix(" °")
        top.addWidget(self.jstep)
        top.addWidget(QLabel("Speed"))
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.01, 3.0); self.speed.setValue(0.6); self.speed.setSuffix(" rad/s")
        top.addWidget(self.speed)
        top.addStretch(1)
        root.addLayout(top)

        # --- joint jogging ---
        jbox = QGroupBox("Joint-space jog")
        jg = QGridLayout(jbox)
        self.joint_labels, self.joint_sliders = [], []
        for i, name in enumerate(_JOINTS):
            jg.addWidget(QLabel(name), i, 0)
            minus = self._jog_button("−")
            minus.clicked.connect(lambda _=False, j=i: self._jog_joint(j, -1))
            plus = self._jog_button("+")
            plus.clicked.connect(lambda _=False, j=i: self._jog_joint(j, +1))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-360, 360)
            slider.sliderReleased.connect(lambda j=i: self._slider_moved(j))
            val = QLabel("0.0°"); val.setMinimumWidth(60)
            val.setStyleSheet("font-family:Consolas")
            jg.addWidget(minus, i, 1)
            jg.addWidget(slider, i, 2)
            jg.addWidget(plus, i, 3)
            jg.addWidget(val, i, 4)
            self.joint_sliders.append(slider)
            self.joint_labels.append(val)
        root.addWidget(jbox)

        # --- cartesian jogging ---
        cbox = QGroupBox("Cartesian jog")
        cg = QGridLayout(cbox)
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Frame"))
        self.frame_box = QComboBox(); self.frame_box.addItems(["Base", "Tool"])
        fr.addWidget(self.frame_box)
        fr.addWidget(QLabel("Lin step"))
        self.lstep = QDoubleSpinBox(); self.lstep.setRange(0.1, 100.0)
        self.lstep.setValue(10.0); self.lstep.setSuffix(" mm")
        fr.addWidget(self.lstep)
        fr.addWidget(QLabel("Rot step"))
        self.rstep = QDoubleSpinBox(); self.rstep.setRange(0.1, 45.0)
        self.rstep.setValue(5.0); self.rstep.setSuffix(" °")
        fr.addWidget(self.rstep)
        fr.addStretch(1)
        cg.addLayout(fr, 0, 0, 1, 4)

        self.cart_labels = []
        for i, name in enumerate(_CART):
            row = i + 1
            cg.addWidget(QLabel(name), row, 0)
            minus = self._jog_button("−")
            minus.clicked.connect(lambda _=False, a=i: self._jog_cart(a, -1))
            plus = self._jog_button("+")
            plus.clicked.connect(lambda _=False, a=i: self._jog_cart(a, +1))
            val = QLabel("0.0"); val.setMinimumWidth(80)
            val.setStyleSheet("font-family:Consolas")
            cg.addWidget(minus, row, 1)
            cg.addWidget(plus, row, 2)
            cg.addWidget(val, row, 3)
            self.cart_labels.append(val)
        root.addWidget(cbox)

        # --- teach waypoint ---
        tbox = QGroupBox("Teach waypoint")
        tg = QHBoxLayout(tbox)
        for label, step in (("MoveJ", "MoveJ"), ("MoveL", "MoveL"),
                            ("MoveP", "MoveP"), ("Process", "ProcessPoint")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, s=step: self.teach_waypoint.emit(s))
            tg.addWidget(b)
        root.addWidget(tbox)

        # home / stop
        hb = QHBoxLayout()
        home = QPushButton("Move Home")
        home.clicked.connect(self._go_home)
        stop = QPushButton("Stop")
        stop.clicked.connect(self._stop)
        hb.addWidget(home); hb.addWidget(stop)
        root.addLayout(hb)
        root.addStretch(1)

    # ---- jog actions ------------------------------------------------------
    @staticmethod
    def _jog_button(text: str) -> QPushButton:
        """A +/- button that repeats while held, like a real teach pendant."""
        b = QPushButton(text)
        b.setFixedWidth(32)
        b.setAutoRepeat(True)
        b.setAutoRepeatDelay(300)       # ms before repeat kicks in
        b.setAutoRepeatInterval(90)     # ms between repeats while held
        return b

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.kin.q_min, self.kin.q_max)

    def _jog_joint(self, joint: int, sign: int) -> None:
        step = np.radians(self.jstep.value())
        target = self._cmd_q.copy()
        target[joint] += sign * step
        self._cmd_q = self._clamp(target)
        self.bridge.move_j(self._cmd_q.tolist(), speed=self.speed.value())
        self._refresh_joint_labels()

    def _slider_moved(self, joint: int) -> None:
        deg = self.joint_sliders[joint].value()
        target = self._cmd_q.copy()
        target[joint] = np.radians(deg)
        self._cmd_q = self._clamp(target)
        self.bridge.move_j(self._cmd_q.tolist(), speed=self.speed.value())
        self._refresh_joint_labels()

    def _jog_cart(self, axis: int, sign: int) -> None:
        from robot.kinematics import rotvec_to_matrix
        pose = self.kin.fk_pose(self._cmd_q)      # jog from the commanded pose
        T = pose_to_matrix(pose)
        delta = np.zeros(6)
        if axis < 3:
            delta[axis] = sign * self.lstep.value() / 1000.0     # mm -> m
        else:
            delta[axis] = sign * np.radians(self.rstep.value())

        if self.frame_box.currentText() == "Tool":
            # increment expressed in the tool frame
            dT = np.eye(4)
            dT[:3, 3] = delta[:3]
            if np.any(delta[3:]):
                dT[:3, :3] = rotvec_to_matrix(delta[3:])
            T_new = T @ dT
        else:
            # base frame: translate directly, rotate about base axes
            T_new = T.copy()
            T_new[:3, 3] = T[:3, 3] + delta[:3]
            if np.any(delta[3:]):
                T_new[:3, :3] = rotvec_to_matrix(delta[3:]) @ T[:3, :3]

        target_pose = matrix_to_pose(T_new)
        res = self.kin.inverse(target_pose, q_init=self._cmd_q)
        # Accept on achieved accuracy rather than the strict success flag —
        # a sub-mm numerical residual must not silently swallow the jog.
        if res.pos_error < 3e-3 and res.rot_error < 3e-2:
            self._cmd_q = self._clamp(res.q)
            # single command: moveJ to the IK solution moves both the real
            # robot and the simulated twin (moveL alone wouldn't update the
            # virtual joint model).
            self.bridge.move_j(self._cmd_q.tolist(), speed=self.speed.value())
        else:
            self.bridge.log.emit(
                "WARN", f"Cartesian jog target unreachable "
                        f"(pos_err={res.pos_error*1000:.1f} mm) — ignored.")
        self._refresh_cart_labels()

    def _go_home(self) -> None:
        home = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        self._cmd_q = home
        self.bridge.move_j(home.tolist(), speed=self.speed.value())

    # ---- units ------------------------------------------------------------
    def _toggle_units(self, use_deg: bool) -> None:
        self._use_deg = use_deg
        self._refresh_joint_labels()

    def _stop(self) -> None:
        """Halt motion: pin the commanded target to the current actual."""
        self._cmd_q = self._actual_q.copy()
        self.bridge.servo_target(self._cmd_q.tolist())

    # ---- state feedback ---------------------------------------------------
    def _on_state(self, st: RobotState) -> None:
        self._actual_q = np.asarray(st.q_actual, float)
        # Re-sync the jog target to the robot when we first see state, while
        # hand-guiding (freedrive), or when the robot has been moved out from
        # under us by more than a large margin (external program, teach).
        diverged = np.linalg.norm(self._actual_q - self._cmd_q) > 0.5
        if not self._synced or st.is_freedrive or diverged:
            self._cmd_q = self._actual_q.copy()
            self._synced = True
        self._refresh_joint_labels()
        self._refresh_cart_labels()

    def _refresh_joint_labels(self) -> None:
        for i, val in enumerate(self.joint_labels):
            q = self._actual_q[i]
            if self._use_deg:
                val.setText(f"{np.degrees(q):+.1f}°")
            else:
                val.setText(f"{q:+.3f}")
            self.joint_sliders[i].blockSignals(True)
            self.joint_sliders[i].setValue(int(np.degrees(q)))
            self.joint_sliders[i].blockSignals(False)

    def _refresh_cart_labels(self) -> None:
        pose = self.kin.fk_pose(self._actual_q)
        for i, val in enumerate(self.cart_labels):
            if i < 3:
                val.setText(f"{pose[i]*1000:+.1f} mm")
            else:
                val.setText(f"{np.degrees(pose[i]):+.1f}°")

    # ---- API for other panels --------------------------------------------
    def current_q(self) -> np.ndarray:
        """Actual joint vector — used when teaching a waypoint."""
        return self._actual_q.copy()

    def current_pose(self) -> np.ndarray:
        return self.kin.fk_pose(self._actual_q)
