"""
ui/panels/step_editor.py
==================================================================
A single modal dialog for editing every parameter of one program
:class:`robot.program.ProgramStep`.

This replaces the old chain of one-value-at-a-time ``QInputDialog``
prompts in the Program Builder. Crucially it lets the user numerically
edit a taught waypoint's **position** — joint angles for a MoveJ, or the
TCP pose (X/Y/Z in mm, Rx/Ry/Rz as rotation-vector components in degrees,
matching the Jog panel's read-outs) for MoveL / MoveP / Process points —
which was previously impossible once a point had been taught.

Units convention (consistent with the rest of the UI)
    * position   : millimetres
    * orientation: rotation-vector components shown in degrees
    * joints     : degrees
Values are converted back to the model's native SI/rad on ``apply``.
==================================================================
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QFormLayout, QGroupBox, QGridLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QPlainTextEdit,
)

from robot.program import ProgramStep, StepType

_JOINTS = ("Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3")
_MOVE_STEPS = {StepType.MOVEJ, StepType.MOVEL, StepType.MOVEP, StepType.PROCESS}
_POSE_STEPS = {StepType.MOVEL, StepType.MOVEP, StepType.PROCESS}


def _spin(value: float, lo: float, hi: float, step: float, decimals: int,
          suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(float(value))
    if suffix:
        sb.setSuffix(suffix)
    sb.setMinimumWidth(110)
    return sb


class StepEditDialog(QDialog):
    """Edit one :class:`ProgramStep` in place. Call :meth:`exec` then, on
    ``Accepted``, :meth:`apply` to write the edited values back onto the step."""

    def __init__(self, step: ProgramStep, parent=None):
        super().__init__(parent)
        self.step = step
        self.setWindowTitle(f"Edit — {step.type.value}")
        self.setModal(True)
        self._joint_spins: List[QDoubleSpinBox] = []
        self._pose_spins: List[QDoubleSpinBox] = []
        self._build()

    # ---- construction -----------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        t = self.step.type

        # optional display-name override (all step types)
        head = QFormLayout()
        self.name_edit = QLineEdit(self.step.name)
        self.name_edit.setPlaceholderText("(optional label — blank = auto)")
        head.addRow("Name", self.name_edit)
        root.addLayout(head)

        if t is StepType.MOVEJ:
            root.addWidget(self._joint_group())
        if t in _POSE_STEPS:
            root.addWidget(self._pose_group())
        if t in _MOVE_STEPS:
            root.addWidget(self._motion_group())
        if t is StepType.DELAY:
            root.addWidget(self._delay_group())
        if t is StepType.SET_DO:
            root.addWidget(self._do_group())
        if t is StepType.WAIT_DI:
            root.addWidget(self._di_group())
        if t is StepType.COMMENT:
            root.addWidget(self._comment_group())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _joint_group(self) -> QGroupBox:
        box = QGroupBox("Joint target (degrees)")
        grid = QGridLayout(box)
        q = self.step.q or [0.0] * 6
        for i, name in enumerate(_JOINTS):
            grid.addWidget(QLabel(name), i, 0)
            sb = _spin(np.degrees(q[i]), -360.0, 360.0, 1.0, 2, " °")
            grid.addWidget(sb, i, 1)
            self._joint_spins.append(sb)
        return box

    def _pose_group(self) -> QGroupBox:
        box = QGroupBox("TCP pose")
        grid = QGridLayout(box)
        pose = self.step.pose or [0.0] * 6
        specs = [
            ("X", pose[0] * 1000.0, -5000.0, 5000.0, 1.0, 2, " mm"),
            ("Y", pose[1] * 1000.0, -5000.0, 5000.0, 1.0, 2, " mm"),
            ("Z", pose[2] * 1000.0, -5000.0, 5000.0, 1.0, 2, " mm"),
            ("Rx", np.degrees(pose[3]), -360.0, 360.0, 1.0, 2, " °"),
            ("Ry", np.degrees(pose[4]), -360.0, 360.0, 1.0, 2, " °"),
            ("Rz", np.degrees(pose[5]), -360.0, 360.0, 1.0, 2, " °"),
        ]
        for i, (name, val, lo, hi, stp, dec, suf) in enumerate(specs):
            grid.addWidget(QLabel(name), i, 0)
            sb = _spin(val, lo, hi, stp, dec, suf)
            grid.addWidget(sb, i, 1)
            self._pose_spins.append(sb)
        return box

    def _motion_group(self) -> QGroupBox:
        box = QGroupBox("Motion parameters")
        form = QFormLayout(box)
        is_joint = self.step.type is StepType.MOVEJ
        v_suffix = " rad/s" if is_joint else " m/s"
        a_suffix = " rad/s²" if is_joint else " m/s²"
        self.speed_spin = _spin(self.step.speed, 0.001, 5.0, 0.05, 3, v_suffix)
        self.accel_spin = _spin(self.step.accel, 0.01, 20.0, 0.1, 3, a_suffix)
        self.blend_spin = _spin(self.step.blend * 1000.0, 0.0, 500.0, 1.0, 1, " mm")
        form.addRow("Speed", self.speed_spin)
        form.addRow("Accel", self.accel_spin)
        form.addRow("Blend radius", self.blend_spin)
        return box

    def _delay_group(self) -> QGroupBox:
        box = QGroupBox("Delay")
        form = QFormLayout(box)
        self.dur_spin = _spin(self.step.duration, 0.0, 3600.0, 0.1, 2, " s")
        form.addRow("Duration", self.dur_spin)
        return box

    def _do_group(self) -> QGroupBox:
        box = QGroupBox("Set digital output")
        form = QFormLayout(box)
        self.pin_spin = QSpinBox(); self.pin_spin.setRange(0, 15)
        self.pin_spin.setValue(int(self.step.pin))
        self.value_chk = QCheckBox("HIGH")
        self.value_chk.setChecked(bool(self.step.value))
        form.addRow("Pin", self.pin_spin)
        form.addRow("State", self.value_chk)
        return box

    def _di_group(self) -> QGroupBox:
        box = QGroupBox("Wait for digital input")
        form = QFormLayout(box)
        self.pin_spin = QSpinBox(); self.pin_spin.setRange(0, 15)
        self.pin_spin.setValue(int(self.step.pin))
        form.addRow("Pin", self.pin_spin)
        return box

    def _comment_group(self) -> QGroupBox:
        box = QGroupBox("Comment")
        v = QVBoxLayout(box)
        self.text_edit = QPlainTextEdit(self.step.text)
        self.text_edit.setFixedHeight(70)
        v.addWidget(self.text_edit)
        return box

    # ---- write back -------------------------------------------------------
    def apply(self) -> None:
        """Persist edited values onto ``self.step`` (call only if accepted)."""
        t = self.step.type
        self.step.name = self.name_edit.text().strip()

        if t is StepType.MOVEJ:
            self.step.q = [float(np.radians(sb.value())) for sb in self._joint_spins]
        if t in _POSE_STEPS:
            p = [sb.value() for sb in self._pose_spins]
            self.step.pose = [
                p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0,
                float(np.radians(p[3])), float(np.radians(p[4])),
                float(np.radians(p[5])),
            ]
        if t in _MOVE_STEPS:
            self.step.speed = self.speed_spin.value()
            self.step.accel = self.accel_spin.value()
            self.step.blend = self.blend_spin.value() / 1000.0
        if t is StepType.DELAY:
            self.step.duration = self.dur_spin.value()
        if t is StepType.SET_DO:
            self.step.pin = int(self.pin_spin.value())
            self.step.value = bool(self.value_chk.isChecked())
        if t is StepType.WAIT_DI:
            self.step.pin = int(self.pin_spin.value())
        if t is StepType.COMMENT:
            self.step.text = self.text_edit.toPlainText().strip()
