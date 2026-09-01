"""
ui/main_window.py
==================================================================
Primary application window — assembles every panel as a dockable
widget around the central 3D digital-twin viewport.

Layout
------
    ┌───────────┬───────────────────────┬────────────┐
    │Connection │                       │  Program   │
    ├───────────┤     3D Viewport       │  Builder   │
    │   Jog     │   (digital twin)      ├────────────┤
    │           │                       │    CAD     │
    ├───────────┴───────────────────────┴────────────┤
    │            Code Editor  (URScript / Python)     │
    └─────────────────────────────────────────────────┘

Threading / rendering
---------------------
Robot feedback arrives on ``bridge.state_updated`` at up to 125 Hz. The
viewport is *not* redrawn per state — a 30 fps ``QTimer`` samples the
latest cached joint vector, keeping VTK smooth and the GUI responsive.
Offline simulation temporarily hijacks that same render loop to play a
planned joint trajectory back on the twin.
==================================================================
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer, QProcess
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QPlainTextEdit, QWidget, QVBoxLayout, QLabel,
    QStatusBar,
)

from robot.ur_bridge import URBridge, RobotState
from robot.kinematics import Kinematics
from robot.program import Program
from robot.ur_models import get_model

from robot.scene import SceneModel
from robot.collision import CollisionWorld, link_radii

from gui.viewport_3d import RobotViewport
from ui.panels.connection_panel import ConnectionPanel
from ui.panels.jog_panel import JogPanel
from ui.panels.program_panel import ProgramPanel
from ui.panels.editor_panel import EditorPanel
from ui.panels.cad_panel import CADPanel
from ui.panels.scene_panel import ScenePanel

log = logging.getLogger("ur_gui.window")


class MainWindow(QMainWindow):
    def __init__(self, bridge: URBridge):
        super().__init__()
        self.bridge = bridge
        self.model = get_model(bridge.config.model_name)
        self.kin = Kinematics(self.model)
        self.program = Program(name="Untitled")
        self.scene = SceneModel(self)
        self._coll_radii = None
        self._project_path: str | None = None      # current .urgproj file, if any

        self.setWindowTitle("UR GUI Simulator")
        self.resize(1500, 950)
        self.setDockNestingEnabled(True)

        # --- central viewport (bridge passed in so joint-dragging can
        #     stream targets to the robot/twin as a software hand-guide) ---
        self.viewport = RobotViewport(self.kin, bridge=bridge)
        self.setCentralWidget(self.viewport)

        # --- panels ---
        self.connection_panel = ConnectionPanel(bridge)
        self.jog_panel = JogPanel(bridge, self.kin)
        self.program_panel = ProgramPanel(bridge, self.program, self.jog_panel, self.kin)
        self.editor_panel = EditorPanel(bridge, self.program)
        self.cad_panel = CADPanel(self.program_panel, self.viewport)
        self.scene_panel = ScenePanel(self.scene, self.viewport,
                                      self.program_panel, self.kin, self)

        self._add_docks()
        self._menu()
        self._status()
        self._connect_signals()

        # --- render / animation loop ---
        self._anim_path: np.ndarray | None = None
        self._anim_index = 0
        self._anim_events: dict = {}
        self._anim_stride = 1
        self._sim_speed = 1.0          # playback speed multiplier (user-set)
        self._carried_local = None
        self._carried_half = None
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start(33)      # ~30 fps

    # ---- docks ------------------------------------------------------------
    def _dock(self, title: str, widget: QWidget, area, minw=320) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setWidget(widget)
        d.setMinimumWidth(minw)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _add_docks(self) -> None:
        L = Qt.DockWidgetArea.LeftDockWidgetArea
        R = Qt.DockWidgetArea.RightDockWidgetArea
        B = Qt.DockWidgetArea.BottomDockWidgetArea

        self.d_conn = self._dock("Connection", self._scroll(self.connection_panel), L)
        self.d_jog = self._dock("Jog / Teach", self._scroll(self.jog_panel), L)
        self.tabifyDockWidget(self.d_conn, self.d_jog)
        self.d_conn.raise_()

        self.d_prog = self._dock("Program Builder", self.program_panel, R)
        self.d_cad = self._dock("CAD / Toolpath", self._scroll(self.cad_panel), R)
        self.d_scene = self._dock("Scene / Palletizer", self._scroll(self.scene_panel), R)
        self.tabifyDockWidget(self.d_prog, self.d_cad)
        self.tabifyDockWidget(self.d_cad, self.d_scene)
        self.d_prog.raise_()

        self.d_edit = self._dock("Code Editor", self.editor_panel, B, minw=400)

    @staticmethod
    def _scroll(widget: QWidget) -> QWidget:
        from PySide6.QtWidgets import QScrollArea
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # ---- menu -------------------------------------------------------------
    def _menu(self) -> None:
        mb = self.menuBar()

        # File: whole-session project save/open (.urgproj) — model + base pose +
        # program + scene, so a user can pick up exactly where they left off.
        file_menu = mb.addMenu("&File")
        for text, seq, fn in (
                ("New Project", QKeySequence.StandardKey.New, self._new_project),
                ("Open Project…", QKeySequence.StandardKey.Open, self._open_project),
                (None, None, None),
                ("Save Project", QKeySequence.StandardKey.Save, self._save_project),
                ("Save Project As…", QKeySequence("Ctrl+Shift+S"),
                 self._save_project_as)):
            if text is None:
                file_menu.addSeparator()
                continue
            act = QAction(text, self)
            act.setShortcut(seq)
            act.triggered.connect(fn)
            file_menu.addAction(act)
            self.addAction(act)

        # Edit: undo/redo/copy/paste for scene objects. Qt shortcuts fire while
        # the app window is active (including when the 3D view has focus), which
        # VTK's own key handling does not do reliably for Ctrl+key combos.
        edit = mb.addMenu("&Edit")
        sp = self.scene_panel
        for text, seq, fn in (
                ("Undo", QKeySequence.StandardKey.Undo, sp.undo),
                ("Redo", QKeySequence.StandardKey.Redo, sp.redo),
                (None, None, None),
                ("Copy object", QKeySequence.StandardKey.Copy, sp.copy_selected),
                ("Paste object", QKeySequence.StandardKey.Paste, sp.paste_clipboard),
                ("Duplicate", QKeySequence("Ctrl+D"), sp.duplicate_selected_scene),
                ("Delete", QKeySequence.StandardKey.Delete, sp.delete_selected_scene)):
            if text is None:
                edit.addSeparator()
                continue
            act = QAction(text, self)
            if text == "Redo":           # accept both Ctrl+Y and Ctrl+Shift+Z
                act.setShortcuts([QKeySequence.StandardKey.Redo,
                                  QKeySequence("Ctrl+Shift+Z")])
            else:
                act.setShortcut(seq)
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            act.triggered.connect(fn)
            edit.addAction(act)
            self.addAction(act)          # ensure the shortcut is app-window active

        view = mb.addMenu("&View")
        for dock in (self.d_conn, self.d_jog, self.d_prog, self.d_cad,
                     self.d_scene, self.d_edit):
            view.addAction(dock.toggleViewAction())
        view.addSeparator()
        reset = QAction("Reset 3D view", self)
        reset.setShortcut(QKeySequence("Ctrl+R"))
        reset.triggered.connect(self.viewport.reset_view)
        view.addAction(reset)

        robot = mb.addMenu("&Robot")
        act_estop = QAction("EMERGENCY STOP", self)
        act_estop.setShortcut(QKeySequence("Esc"))
        act_estop.triggered.connect(self.bridge.emergency_stop)
        robot.addAction(act_estop)
        robot.addSeparator()
        self.act_download = QAction("Download UR CAD meshes for this model", self)
        self.act_download.triggered.connect(self._download_meshes)
        robot.addAction(self.act_download)

        help_menu = mb.addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _about(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "UR GUI Simulator",
            "<b>UR GUI Simulator</b><br>"
            "Digital twin, teach pendant, CAD toolpaths and dual-mode "
            "code generation for Universal Robots.<br><br>"
            f"RTDE backend: {'available' if URBridge.rtde_available() else 'socket/sim fallback'}")

    # ---- project save / open ---------------------------------------------
    def _gather_project(self) -> str:
        from robot.project import project_to_json
        return project_to_json(self.model.name, self.kin.base_pose(),
                               self.program, self.scene)

    def _apply_project(self, data: dict) -> None:
        """Restore a loaded project onto the *live* objects the panels share
        (so nothing needs to be rebuilt). Order matters: set the model first
        (which resets kinematics), load the program and scene, then restore the
        exact base pose last so a pedestal mount / repositioning survives."""
        from robot.program import Program
        from robot.ur_models import MODEL_NAMES
        # 1) robot model — setting the combo fires _on_model_changed, which
        #    rebuilds kinematics + viewport for the new arm.
        name = data.get("model")
        if name in MODEL_NAMES and name != self.model.name:
            self.connection_panel.model_box.setCurrentText(name)
        # 2) program, mutated in place so program_panel/editor_panel keep working.
        self.program.load_from(Program.from_dict(data.get("program", {})))
        self.program_panel.refresh()
        self.program_panel.program_changed.emit()
        # 3) scene (emits changed → scene_panel refresh + pedestal base-height sync).
        self.scene.load_dict(data.get("scene", {}))
        # 4) exact base pose (pedestal height + any drag/rotate) — after the scene
        #    load, whose auto base-height sync would otherwise clobber the z.
        bp = data.get("base_pose")
        if bp is not None:
            self.kin.set_base_pose(np.asarray(bp, float))
            q = getattr(self.viewport, "_q", None)
            if q is not None:
                self.viewport.update_joints(q)          # re-pose the twin, lifted
            self.scene_panel._rebuild_axes()

    def _new_project(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from robot.program import Program
        if QMessageBox.question(
                self, "New Project",
                "Start a new project? Any unsaved changes will be lost.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.program.load_from(Program(name="Untitled"))
        self.program_panel.refresh()
        self.program_panel.program_changed.emit()
        self.scene.clear()
        self._project_path = None
        self._update_title()
        self._set_status("New project.")

    def _open_project(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from robot.project import project_from_json, EXTENSION
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", f"UR GUI project (*{EXTENSION})")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = project_from_json(fh.read())
            self._apply_project(data)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Open Project", f"Could not open:\n{exc}")
            return
        self._project_path = path
        self._update_title()
        self._set_status(f"Opened project {path}")

    def _save_project(self) -> None:
        if self._project_path is None:
            self._save_project_as()
            return
        self._write_project(self._project_path)

    def _save_project_as(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from robot.project import EXTENSION
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", f"UR GUI project (*{EXTENSION})")
        if not path:
            return
        if not path.lower().endswith(EXTENSION):
            path += EXTENSION
        self._write_project(path)

    def _write_project(self, path: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._gather_project())
        except OSError as exc:
            QMessageBox.warning(self, "Save Project", f"Could not save:\n{exc}")
            return
        self._project_path = path
        self._update_title()
        self._set_status(f"Saved project {path}")

    def _update_title(self) -> None:
        name = os.path.basename(self._project_path) if self._project_path else None
        self.setWindowTitle("UR GUI Simulator" + (f" — {name}" if name else ""))

    # ---- status bar + log -------------------------------------------------
    def _status(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_lbl = QLabel("Ready.")
        sb.addWidget(self._status_lbl, 1)

        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setFixedHeight(120)
        self.log_view.setStyleSheet("font-family:Consolas;font-size:10px")
        d = QDockWidget("Log", self)
        d.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, d)
        self.d_log = d
        self.tabifyDockWidget(self.d_edit, d)
        self.d_edit.raise_()

    # ---- signals ----------------------------------------------------------
    def _connect_signals(self) -> None:
        self.bridge.log.connect(self._on_log)
        self.connection_panel.model_changed.connect(self._on_model_changed)
        self.jog_panel.teach_waypoint.connect(self.program_panel.add_taught)
        self.program_panel.program_changed.connect(self.editor_panel.regenerate)
        self.program_panel.simulate_requested.connect(self._start_animation)
        self.program_panel.status.connect(self._set_status)
        self.cad_panel.toolpath_ready.connect(
            lambda poses: self._set_status(f"Toolpath ready: {len(poses)} points."))
        # (scene box rendering + selection highlight is owned by scene_panel)
        self.program_panel.set_collision_checker(self._collision_check)
        # A pedestal under the robot mounts it higher: keep the base height in
        # sync with the pedestal stack whenever the scene is edited.
        self.scene.changed.connect(self._sync_base_height)

    def _sync_base_height(self) -> None:
        """Lift the robot base to the top of the pedestal stack (0 if none)."""
        h = self.scene.pedestal_height()
        if abs(h - getattr(self.kin, "base_height", 0.0)) < 1e-9:
            return                                      # unchanged — no re-render
        self.kin.set_base_height(h)
        self.viewport.update_joints(self.viewport._q)   # re-pose the twin, lifted

    def _collision_check(self, q_path):
        """Scene collision hook for offline simulation. Returns (idx, result)."""
        if self._coll_radii is None:
            self._coll_radii = link_radii(self.kin)
        boxes = self.scene.all_collision_boxes()
        if not boxes:
            return -1, None
        world = CollisionWorld(boxes, self._coll_radii)
        return world.first_collision(self.kin, q_path, margin=0.005)

    def _on_model_changed(self, name: str) -> None:
        self.model = get_model(name)
        self.kin.set_model(self.model)
        self.kin.set_base_height(self.scene.pedestal_height())  # re-apply mount
        self.viewport.set_kinematics(self.kin)
        self._coll_radii = None
        self.scene_panel._radii = None
        self.scene_panel.apply_model_defaults()
        self.scene_panel._rebuild_axes()      # robot base reset → redraw gizmos
        self._last_render_q = None
        self._set_status(f"Model set to {name}. "
                         + ("Real UR CAD loaded." if self.viewport._using_meshes
                            else "Using stick model — Robot ▸ Download UR CAD meshes."))

    # ---- mesh download ----------------------------------------------------
    def _download_meshes(self) -> None:
        name = self.model.name
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "scripts", "fetch_ur_meshes.py")
        self.act_download.setEnabled(False)
        self._set_status(f"Downloading UR CAD meshes for {name}…")
        self._dl = QProcess(self)
        self._dl.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._dl.readyReadStandardOutput.connect(
            lambda: self._on_log("INFO", bytes(
                self._dl.readAllStandardOutput()).decode(errors="replace").strip()))
        self._dl.finished.connect(lambda *_: self._on_meshes_ready(name))
        self._dl.start(sys.executable, [script, name])

    def _on_meshes_ready(self, name: str) -> None:
        self.act_download.setEnabled(True)
        self.viewport.set_kinematics(self.kin)     # rebuild twin with real CAD
        self._last_render_q = None
        ok = self.viewport._using_meshes
        self._set_status(f"{name}: real UR CAD loaded." if ok
                         else f"{name}: mesh download finished but meshes not found.")

    def _on_log(self, level: str, msg: str) -> None:
        self.log_view.appendPlainText(f"{level:5s} | {msg}")
        if level in ("WARN", "ERROR"):
            self._set_status(msg)

    def _set_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    # ---- render / animation ----------------------------------------------
    @staticmethod
    def _playback_stride(n: int) -> int:
        # keep 1x playback to roughly 450 frames (~15 s at 30 fps)
        return max(1, n // 450)

    def set_sim_speed(self, factor: float) -> None:
        """Set the animation playback speed multiplier (0 ⇒ jump to the end).

        Applied live in :meth:`_render_tick`, so it takes effect mid-playback —
        handy for fast-forwarding a long palletizing run while testing.
        """
        self._sim_speed = max(0.0, float(factor))

    def _effective_stride(self) -> int:
        if self._sim_speed <= 0.0:                 # "Instant"
            return max(1, len(self._anim_path)) if self._anim_path is not None else 1
        return max(1, int(round(self._anim_stride * self._sim_speed)))

    def _start_animation(self, q_path: np.ndarray) -> None:
        self._anim_path = np.asarray(q_path)
        self._anim_index = 0
        self._anim_events = {}
        self._anim_stride = self._playback_stride(len(self._anim_path))
        self._carried_local = None
        self._carried_half = None
        self.viewport.set_carried_box(None)

    def play_job(self, q_path: np.ndarray, events: dict,
                 initial_visible=None) -> None:
        """Animate a palletizing/transfer job: joint path + timeline events
        (reveal or hide a box, attach/detach the carried box). ``initial_visible``
        seeds which placed boxes start shown (a depalletize source pallet); None
        starts with everything hidden (a plain palletize)."""
        self._anim_path = np.asarray(q_path)
        self._anim_index = 0
        self._anim_events = dict(events or {})
        self._anim_stride = self._playback_stride(len(self._anim_path))
        self._carried_local = None
        self._carried_half = None
        self.viewport.set_carried_box(None)
        if initial_visible is None:
            self.viewport.set_placed_visible(0)
        else:
            self.viewport.set_boxes_visible(initial_visible)

    def _apply_anim_event(self, i: int) -> None:
        ev = self._anim_events.get(i)
        if ev is None:
            return
        if ev[0] == "carry":
            self._carried_local = np.asarray(ev[1], float)
            self._carried_half = np.asarray(ev[2], float)
        elif ev[0] == "drop_reveal":
            self._carried_local = None
            self.viewport.set_carried_box(None)
            self.viewport.set_placed_visible(int(ev[1]))
        elif ev[0] == "box_show":               # place a depalletized box
            self._carried_local = None
            self.viewport.set_carried_box(None)
            self.viewport.set_box_visible(int(ev[1]), True)
        elif ev[0] == "box_hide":               # source box just picked up
            self.viewport.set_box_visible(int(ev[1]), False)

    def _render_tick(self) -> None:
        # While the user is hand-guiding a joint in the 3D view, the drag
        # owns the viewport — don't overwrite it with (lagging) feedback.
        if getattr(self.viewport, "dragging", False):
            return
        if self._anim_path is not None:
            n = len(self._anim_path)
            stride = self._effective_stride()
            i = self._anim_index
            # apply every timeline event in the frames we're about to skip over
            for j in range(i, min(i + stride, n)):
                self._apply_anim_event(j)
            k = min(i + stride - 1, n - 1)
            q = self._anim_path[k]
            self.viewport.update_joints(q)
            if self._carried_local is not None:
                T = self.kin.fk_frames(np.asarray(q, float))[-1] @ self._carried_local
                self.viewport.set_carried_box(T, self._carried_half)
            self._anim_index += stride
            if self._anim_index >= n:
                self._anim_path = None
                self._anim_events = {}
                self._carried_local = None
                self.viewport.set_carried_box(None)
                self._set_status("Simulation complete.")
            return
        # live: follow latest cached joint state — but only actually re-pose
        # the twin when the joints changed. Constantly re-rendering an idle
        # robot competes with camera interaction and makes it feel stuttery.
        st: RobotState = self.bridge.state
        q = np.asarray(st.q_actual, float)
        last = getattr(self, "_last_render_q", None)
        if last is None or not np.allclose(q, last, atol=2e-4):
            self.viewport.update_joints(q)
            self._last_render_q = q.copy()
            if st.transport.value != "RTDE":
                self.bridge.set_tcp_pose(self.kin.fk_pose(q).tolist())

    # ---- shutdown ---------------------------------------------------------
    def closeEvent(self, event):          # noqa: N802 (Qt override)
        self._render_timer.stop()
        self.bridge.shutdown()
        super().closeEvent(event)
