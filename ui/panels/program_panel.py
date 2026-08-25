"""
ui/panels/program_panel.py
==================================================================
Interactive Program Builder dock.

A :class:`robot.program.Program` rendered as a reorderable tree:

    * Teach Waypoint  -> MoveJ / MoveL / MoveP / Process point
    * Insert  Gripper Open/Close, Delay, Set DO, Wait DI, Comment
    * Reorder (up/down), enable/disable, edit params, delete
    * Save / Load (.urgui JSON)
    * Run Offline Simulation  (animate the digital twin, collision-free check)
    * Execute on Robot        (with a safety confirmation)

The panel owns the shared Program instance and emits ``program_changed``
whenever it mutates so the code editor can regenerate scripts live.
==================================================================
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton,
    QGroupBox, QGridLayout, QInputDialog, QFileDialog, QMessageBox, QMenu,
    QDoubleSpinBox, QLabel,
)
from PySide6.QtGui import QColor, QBrush

from robot.program import Program, ProgramStep, StepType
from robot.ur_bridge import URBridge

_UID_ROLE = Qt.ItemDataRole.UserRole


class ProgramPanel(QWidget):
    program_changed = Signal()
    simulate_requested = Signal(object)     # emits np.ndarray q_path
    status = Signal(str)

    def __init__(self, bridge: URBridge, program: Program, jog_panel, kin, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.program = program
        self.jog = jog_panel
        self.kin = kin
        self._collision_checker = None
        self._build()
        self.refresh()

    def set_collision_checker(self, fn) -> None:
        """fn(q_path) -> (index, CollisionResult|None); index<0 means clear."""
        self._collision_checker = fn

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        # insert buttons
        add = QGroupBox("Add step")
        g = QGridLayout(add)
        actions = [
            ("MoveJ", lambda: self._teach(StepType.MOVEJ)),
            ("MoveL", lambda: self._teach(StepType.MOVEL)),
            ("MoveP", lambda: self._teach(StepType.MOVEP)),
            ("Process Pt", lambda: self._teach(StepType.PROCESS)),
            ("Gripper Open", lambda: self._add(ProgramStep(StepType.GRIPPER_OPEN))),
            ("Gripper Close", lambda: self._add(ProgramStep(StepType.GRIPPER_CLOSE))),
            ("Delay", self._add_delay),
            ("Set DO", self._add_do),
            ("Wait DI", self._add_wait_di),
            ("Comment", self._add_comment),
        ]
        for i, (label, fn) in enumerate(actions):
            b = QPushButton(label)
            b.clicked.connect(fn)
            g.addWidget(b, i // 2, i % 2)
        root.addWidget(add)

        # tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Step", "Speed", "Accel"])
        self.tree.setColumnWidth(0, 34)
        self.tree.setColumnWidth(1, 260)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemDoubleClicked.connect(lambda *_: self._edit_selected())
        root.addWidget(self.tree, 1)

        # edit controls
        edit = QHBoxLayout()
        for label, fn in (("▲", lambda: self._reorder(-1)),
                          ("▼", lambda: self._reorder(+1)),
                          ("Edit", self._edit_selected),
                          ("Enable/Disable", self._toggle_enabled),
                          ("Delete", self._delete_selected)):
            b = QPushButton(label); b.clicked.connect(fn)
            edit.addWidget(b)
        root.addLayout(edit)

        # file + run
        fr = QHBoxLayout()
        save = QPushButton("Save…"); save.clicked.connect(self._save)
        load = QPushButton("Load…"); load.clicked.connect(self._load)
        fr.addWidget(save); fr.addWidget(load)
        root.addLayout(fr)

        run = QHBoxLayout()
        self.sim_steps = QDoubleSpinBox(); self.sim_steps.setRange(5, 200)
        self.sim_steps.setValue(40); self.sim_steps.setPrefix("steps ")
        sim = QPushButton("Run Offline Simulation")
        sim.clicked.connect(self._simulate)
        exe = QPushButton("Execute on Robot")
        exe.setStyleSheet("background:#a3be8c;font-weight:bold")
        exe.clicked.connect(self._execute)
        run.addWidget(self.sim_steps)
        run.addWidget(sim, 1)
        run.addWidget(exe, 1)
        root.addLayout(run)

    # ---- add / teach ------------------------------------------------------
    def _teach(self, step_type: StepType) -> None:
        q = self.jog.current_q()
        pose = self.jog.current_pose()
        if step_type is StepType.MOVEJ:
            step = ProgramStep(step_type, q=q.tolist(),
                               speed=self.program.default_j_speed,
                               accel=self.program.default_j_accel)
        else:
            step = ProgramStep(step_type, pose=pose.tolist(),
                               speed=self.program.default_l_speed,
                               accel=self.program.default_l_accel)
        self._add(step)

    def add_taught(self, step_type_value: str) -> None:
        """Called by MainWindow when the Jog panel's Teach button fires."""
        self._teach(StepType(step_type_value))

    def add_toolpath(self, poses: np.ndarray, speed: float = 0.1,
                     blend: float = 0.002) -> None:
        """Append a CAD-derived toolpath as a sequence of Process points."""
        for p in poses:
            self.program.add(ProgramStep(StepType.PROCESS, pose=list(p),
                                         speed=speed, accel=0.5, blend=blend))
        self.refresh()
        self.program_changed.emit()
        self.status.emit(f"Added {len(poses)} process points from toolpath.")

    def add_program_steps(self, steps) -> None:
        """Append pre-built ProgramStep objects (e.g. from the Path Editor)."""
        steps = list(steps)
        if not steps:
            return
        self.program.steps.extend(steps)
        self.refresh()
        self.program_changed.emit()
        self.status.emit(f"Added {len(steps)} step(s) from Path Editor.")

    def _add(self, step: ProgramStep) -> None:
        idx = self._selected_index()
        self.program.add(step, index=None if idx is None else idx + 1)
        self.refresh()
        self.program_changed.emit()

    def _add_delay(self) -> None:
        val, ok = QInputDialog.getDouble(self, "Delay", "Seconds:", 1.0, 0.0, 3600, 2)
        if ok:
            self._add(ProgramStep(StepType.DELAY, duration=val))

    def _add_do(self) -> None:
        pin, ok = QInputDialog.getInt(self, "Set Digital Out", "Pin:", 0, 0, 15)
        if not ok:
            return
        val = QMessageBox.question(self, "Set Digital Out", f"Set DO[{pin}] HIGH?") \
            == QMessageBox.StandardButton.Yes
        self._add(ProgramStep(StepType.SET_DO, pin=pin, value=val))

    def _add_wait_di(self) -> None:
        pin, ok = QInputDialog.getInt(self, "Wait Digital In", "Pin:", 0, 0, 15)
        if ok:
            self._add(ProgramStep(StepType.WAIT_DI, pin=pin))

    def _add_comment(self) -> None:
        text, ok = QInputDialog.getText(self, "Comment", "Text:")
        if ok:
            self._add(ProgramStep(StepType.COMMENT, text=text))

    # ---- selection helpers ------------------------------------------------
    def _selected_uid(self):
        items = self.tree.selectedItems()
        return items[0].data(0, _UID_ROLE) if items else None

    def _selected_index(self):
        uid = self._selected_uid()
        return self.program._index(uid) if uid is not None else None

    def _selected_step(self):
        idx = self._selected_index()
        return self.program.steps[idx] if idx is not None else None

    # ---- edit ops ---------------------------------------------------------
    def _reorder(self, delta: int) -> None:
        uid = self._selected_uid()
        if uid is not None:
            self.program.move(uid, delta)
            self.refresh(select_uid=uid)
            self.program_changed.emit()

    def _delete_selected(self) -> None:
        uid = self._selected_uid()
        if uid is not None:
            self.program.remove(uid)
            self.refresh()
            self.program_changed.emit()

    def _toggle_enabled(self) -> None:
        step = self._selected_step()
        if step:
            step.enabled = not step.enabled
            self.refresh(select_uid=step.uid)
            self.program_changed.emit()

    def _edit_selected(self) -> None:
        step = self._selected_step()
        if not step:
            return
        if step.type in (StepType.MOVEJ, StepType.MOVEL, StepType.MOVEP,
                         StepType.PROCESS):
            v, ok = QInputDialog.getDouble(self, "Edit speed", "Speed:",
                                           step.speed, 0.001, 5.0, 3)
            if ok:
                step.speed = v
            v, ok = QInputDialog.getDouble(self, "Edit accel", "Accel:",
                                           step.accel, 0.01, 10.0, 3)
            if ok:
                step.accel = v
            v, ok = QInputDialog.getDouble(self, "Blend radius (m)", "Blend:",
                                           step.blend, 0.0, 0.5, 4)
            if ok:
                step.blend = v
        elif step.type is StepType.DELAY:
            v, ok = QInputDialog.getDouble(self, "Delay", "Seconds:",
                                           step.duration, 0.0, 3600, 2)
            if ok:
                step.duration = v
        elif step.type is StepType.COMMENT:
            v, ok = QInputDialog.getText(self, "Comment", "Text:", text=step.text)
            if ok:
                step.text = v
        self.refresh(select_uid=step.uid)
        self.program_changed.emit()

    def _context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Edit", self._edit_selected)
        menu.addAction("Move up", lambda: self._reorder(-1))
        menu.addAction("Move down", lambda: self._reorder(+1))
        menu.addAction("Enable/Disable", self._toggle_enabled)
        menu.addSeparator()
        menu.addAction("Delete", self._delete_selected)
        menu.exec(self.tree.mapToGlobal(pos))

    # ---- persistence ------------------------------------------------------
    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save program", "",
                                              "UR GUI program (*.urgui)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.program.to_json())
            self.status.emit(f"Saved {path}")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load program", "",
                                              "UR GUI program (*.urgui)")
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            loaded = Program.from_json(fh.read())
        self.program.name = loaded.name
        self.program.steps = loaded.steps
        self.program.tcp = loaded.tcp
        self.refresh()
        self.program_changed.emit()
        self.status.emit(f"Loaded {path}")

    # ---- simulate / execute ----------------------------------------------
    def _build_joint_path(self) -> np.ndarray:
        """Flatten the program's motion steps into one dense joint trajectory."""
        from robot.kinematics import TrajectoryPlanner
        planner = TrajectoryPlanner(self.kin)
        steps = int(self.sim_steps.value())
        q = self.jog.current_q()
        path = [q]
        for s in self.program.steps:
            if not s.enabled:
                continue
            if s.type is StepType.MOVEJ and s.q is not None:
                seg = planner.joint_move(path[-1], np.asarray(s.q), steps)
                path.extend(seg[1:])
            elif s.type in (StepType.MOVEL, StepType.MOVEP, StepType.PROCESS) \
                    and s.pose is not None:
                target_q = self.kin.inverse(np.asarray(s.pose), q_init=path[-1]).q
                seg = planner.joint_move(path[-1], target_q, steps)
                path.extend(seg[1:])
        return np.asarray(path)

    def _simulate(self) -> None:
        if not any(s.enabled for s in self.program.steps):
            self.status.emit("Program is empty — nothing to simulate.")
            return
        path = self._build_joint_path()
        # simple self-limit / reachability check
        bad = sum(1 for q in path if not self.kin.in_limits(q))
        if bad:
            self.status.emit(f"⚠ {bad} sample(s) exceed joint limits.")
        # collision check against the scene, if one is wired in
        if self._collision_checker is not None:
            try:
                idx, res = self._collision_checker(path)
            except Exception:                              # noqa: BLE001
                idx, res = -1, None
            if idx is not None and idx >= 0 and res is not None:
                self.status.emit(
                    f"⚠ Collision at sample {idx}/{len(path)}: "
                    f"{res.link} ↔ {res.box}.")
        self.simulate_requested.emit(path)
        self.status.emit(f"Simulating {len(path)} trajectory samples…")

    def _execute(self) -> None:
        if not any(s.enabled for s in self.program.steps):
            self.status.emit("Program is empty — nothing to execute.")
            return
        reply = QMessageBox.warning(
            self, "Execute on robot",
            "This will move the PHYSICAL robot.\n\n"
            "Ensure the area is clear and you can reach the E-Stop.\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Always animate the digital twin so the operator sees what will run.
        try:
            self.simulate_requested.emit(self._build_joint_path())
        except Exception:                                  # noqa: BLE001
            pass
        # Sequencing matters: firing per-step queued moves would let each
        # command override the previous one on the controller. Instead we
        # compile the whole program to a single URScript unit and send it
        # once — the controller then executes the steps in order (honouring
        # blend radii). In simulation this is logged; on a real robot it runs.
        from robot.program import URScriptGenerator
        script = URScriptGenerator().generate(self.program)
        self.bridge.run_script(script)
        self.status.emit(
            f"Program ({sum(s.enabled for s in self.program.steps)} steps) "
            "sent to robot as one URScript unit.")

    # ---- render -----------------------------------------------------------
    def refresh(self, select_uid=None) -> None:
        self.tree.clear()
        for i, s in enumerate(self.program.steps):
            item = QTreeWidgetItem([
                str(i + 1), s.label(),
                f"{s.speed:g}" if s.type.value.startswith("Move") or s.type == StepType.PROCESS else "",
                f"{s.accel:g}" if s.type.value.startswith("Move") or s.type == StepType.PROCESS else "",
            ])
            item.setData(0, _UID_ROLE, s.uid)
            if not s.enabled:
                for c in range(4):
                    item.setForeground(c, QBrush(QColor("#6c7480")))
            self.tree.addTopLevelItem(item)
            if s.uid == select_uid:
                self.tree.setCurrentItem(item)
