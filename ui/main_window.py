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

from gui.viewport_3d import RobotViewport
from ui.panels.connection_panel import ConnectionPanel
from ui.panels.jog_panel import JogPanel
from ui.panels.program_panel import ProgramPanel
from ui.panels.editor_panel import EditorPanel
from ui.panels.cad_panel import CADPanel

log = logging.getLogger("ur_gui.window")


class MainWindow(QMainWindow):
    def __init__(self, bridge: URBridge):
        super().__init__()
        self.bridge = bridge
        self.model = get_model(bridge.config.model_name)
        self.kin = Kinematics(self.model)
        self.program = Program(name="Untitled")

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

        self._add_docks()
        self._menu()
        self._status()
        self._connect_signals()

        # --- render / animation loop ---
        self._anim_path: np.ndarray | None = None
        self._anim_index = 0
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
        self.tabifyDockWidget(self.d_prog, self.d_cad)
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
        view = mb.addMenu("&View")
        for dock in (self.d_conn, self.d_jog, self.d_prog, self.d_cad, self.d_edit):
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

    def _on_model_changed(self, name: str) -> None:
        self.model = get_model(name)
        self.kin.set_model(self.model)
        self.viewport.set_kinematics(self.kin)
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
    def _start_animation(self, q_path: np.ndarray) -> None:
        self._anim_path = np.asarray(q_path)
        self._anim_index = 0

    def _render_tick(self) -> None:
        # While the user is hand-guiding a joint in the 3D view, the drag
        # owns the viewport — don't overwrite it with (lagging) feedback.
        if getattr(self.viewport, "dragging", False):
            return
        if self._anim_path is not None:
            q = self._anim_path[self._anim_index]
            self.viewport.update_joints(q)
            self._anim_index += 1
            if self._anim_index >= len(self._anim_path):
                self._anim_path = None
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
