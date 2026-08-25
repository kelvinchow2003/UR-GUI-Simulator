"""
ui/panels/path_editor.py
==================================================================
Interactive **Path Editor** — draw a robot toolpath by clicking on an
imported CAD part (and in free space around it), then export it.

Launched from the CAD dock's *"Create Path with CAD"* button, this opens
a dedicated full window containing its own digital-twin viewport
(:class:`gui.viewport_3d.RobotViewport`) so the main window is never
disturbed. The user:

    * clicks to drop waypoints — a **surface pick** snaps the point onto
      the CAD; a miss falls back to an adjustable **work plane** so points
      can start away from the part and pan into it;
    * drags points, deletes, inserts, closes the loop, reverses, undo/redo;
    * tunes move type (MoveJ/MoveL/MoveP/Process), speed, approach
      direction, tool standoff and lead-in/out;
    * previews the path live on the robot with orientation triads and
      instant reachability feedback (unreachable points turn red);
    * exports as a ``.script`` file, into the main Program, or to clipboard.

Coordinate frame
----------------
The robot base is the world origin, so every picked VTK world point is
already expressed **relative to the centre of the robot's base** — exactly
the convention the rest of the app (and UR's ``movel``) uses: position in
metres, orientation as an axis-angle rotation vector.
==================================================================
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QBrush, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QTreeWidget,
    QTreeWidgetItem, QFileDialog, QMessageBox, QDockWidget, QScrollArea,
    QToolBar, QApplication,
)

from gui.viewport_3d import RobotViewport, _HAVE_PV
from cad.cad_importer import ToolpathGenerator
from robot.kinematics import Kinematics, matrix_to_pose
from robot.program import Program, ProgramStep, StepType, URScriptGenerator

log = logging.getLogger("ur_gui.pathedit")

# approach-direction menu -> unit vector in the base frame
_APPROACH = {
    "-Z  (down onto part)": (0.0, 0.0, -1.0),
    "+Z  (up)": (0.0, 0.0, 1.0),
    "-X": (-1.0, 0.0, 0.0),
    "+X": (1.0, 0.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
}
_MOVE_TYPES = {
    "MoveL": StepType.MOVEL,
    "MoveP": StepType.MOVEP,
    "Process": StepType.PROCESS,
    "MoveJ": StepType.MOVEJ,
}
_TOOL_SELECT, _TOOL_ADD, _TOOL_MOVE = "select", "add", "move"


# ===========================================================================
#  Waypoint model
# ===========================================================================
@dataclass
class Waypoint:
    """One picked point on (or around) the CAD, in base-frame metres."""
    pos: np.ndarray                              # (3,)
    move: str = "MoveL"                          # MoveJ/MoveL/MoveP/Process
    on_surface: bool = False                     # picked on the CAD mesh
    normal: Optional[np.ndarray] = None          # surface normal (base frame)
    reachable: bool = True                       # last IK check (for colouring)


# ===========================================================================
#  VTK draw controller — mirrors gui.viewport_3d._DragController
# ===========================================================================
class _PathDrawController:
    """
    Mouse observers on the editor viewport's interactor.

    * a clean left-click (no drag) drops / selects / does nothing per tool;
    * a left-drag orbits the camera as usual (so the trackball still works);
    * the Move tool freezes the camera while dragging the picked waypoint.

    Ctrl/Shift + drag is deliberately ignored here so the viewport's own
    joint hand-guide keeps working inside the editor too.
    """

    _CLICK_PX = 6          # press/release move under this = a click, not a drag

    def __init__(self, window: "PathEditorWindow", viewport: RobotViewport, iren):
        import vtk                                # noqa: WPS433 (lazy)
        self._vtk = vtk
        self._win = window
        self._vp = viewport
        self._iren = iren
        self._null_style = vtk.vtkInteractorStyleUser()
        self._saved_style = None
        self._press_xy = (0, 0)
        self._moving = False           # dragging a waypoint (Move tool)
        self._maybe_click = False
        iren.AddObserver("LeftButtonPressEvent", self._on_press, 10.0)
        iren.AddObserver("MouseMoveEvent", self._on_move, 10.0)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_release, 10.0)

    # ---- geometry: click -> 3D point -------------------------------------
    def _pick_point(self):
        """(pos, normal|None, on_surface) for the current cursor position."""
        x, y = self._iren.GetEventPosition()
        ren = self._vp.plotter.renderer
        cad = self._vp.cad_actor()
        # 1) try the CAD surface (unless the user turned snapping off)
        if cad is not None and self._win.snap_enabled():
            try:
                picker = self._vtk.vtkCellPicker()
                picker.SetTolerance(0.0008)
                picker.PickFromListOn()
                picker.InitializePickList()
                picker.AddPickList(cad)
                if picker.Pick(x, y, 0, ren) and picker.GetCellId() != -1:
                    pos = np.array(picker.GetPickPosition(), float)
                    n = np.array(picker.GetPickNormal(), float)
                    if np.linalg.norm(n) < 1e-6:
                        n = None
                    return pos, n, True
            except Exception:                      # noqa: BLE001
                pass
        # 2) fall back to the work plane
        p0, p1 = self._ray(x, y, ren)
        pt, nrm = self._win.work_plane()
        d = p1 - p0
        denom = float(np.dot(d, nrm))
        if abs(denom) < 1e-9:
            return None, None, False
        t = float(np.dot(pt - p0, nrm) / denom)
        return p0 + t * d, None, False

    @staticmethod
    def _ray(x, y, ren):
        """Two world points along the click ray (near/far), for plane hits."""
        ren.SetDisplayPoint(x, y, 0.0); ren.DisplayToWorld()
        w0 = np.array(ren.GetWorldPoint(), float)
        ren.SetDisplayPoint(x, y, 1.0); ren.DisplayToWorld()
        w1 = np.array(ren.GetWorldPoint(), float)
        p0 = w0[:3] / (w0[3] if abs(w0[3]) > 1e-12 else 1.0)
        p1 = w1[:3] / (w1[3] if abs(w1[3]) > 1e-12 else 1.0)
        return p0, p1

    # ---- observers --------------------------------------------------------
    def _on_press(self, obj, evt):
        if self._iren.GetControlKey() or self._iren.GetShiftKey():
            return                                 # let joint hand-guide own it
        self._press_xy = self._iren.GetEventPosition()
        tool = self._win.current_tool()
        if tool == _TOOL_MOVE:
            idx = self._win.nearest_waypoint(*self._press_xy)
            if idx is not None:
                self._win.select_index(idx)
                self._moving = True
                self._freeze_camera()
            return
        # add / select are decided on release (click vs drag)
        self._maybe_click = True

    def _on_move(self, obj, evt):
        if self._moving:
            pos, n, on_surf = self._pick_point()
            if pos is not None:
                self._win.update_waypoint(self._win.selected, pos, n, on_surf)

    def _on_release(self, obj, evt):
        if self._moving:
            self._moving = False
            self._thaw_camera()
            self._win.commit_edit()
            return
        if not self._maybe_click:
            return
        self._maybe_click = False
        x, y = self._iren.GetEventPosition()
        if abs(x - self._press_xy[0]) > self._CLICK_PX or \
           abs(y - self._press_xy[1]) > self._CLICK_PX:
            return                                 # it was a camera orbit
        tool = self._win.current_tool()
        if tool == _TOOL_ADD:
            pos, n, on_surf = self._pick_point()
            if pos is not None:
                self._win.add_waypoint(pos, n, on_surf)
        elif tool == _TOOL_SELECT:
            idx = self._win.nearest_waypoint(x, y)
            if idx is not None:
                self._win.select_index(idx)

    # ---- camera freeze (identical trick to _DragController) --------------
    def _freeze_camera(self):
        self._saved_style = self._iren.GetInteractorStyle()
        self._iren.SetInteractorStyle(self._null_style)

    def _thaw_camera(self):
        if self._saved_style is not None:
            self._iren.SetInteractorStyle(self._saved_style)
            try:
                self._saved_style.StopState()
            except Exception:                      # noqa: BLE001
                pass
            self._saved_style = None


# ===========================================================================
#  Path Editor window
# ===========================================================================
class PathEditorWindow(QMainWindow):
    def __init__(self, kin: Kinematics, cad, T_base_cad: np.ndarray,
                 program_panel, q0=None, parent=None):
        super().__init__(parent)
        self.kin = kin
        self.cad = cad
        self.T_base_cad = np.asarray(T_base_cad, float) if T_base_cad is not None \
            else np.eye(4)
        self.program_panel = program_panel
        self.q0 = np.asarray(q0, float) if q0 is not None \
            else np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

        self.waypoints: List[Waypoint] = []
        self.selected: int = -1
        self._undo: List = []
        self._redo: List = []
        self._tool = _TOOL_ADD

        self.setWindowTitle("Path Editor — draw a toolpath on the CAD")
        self.resize(1400, 900)

        if not _HAVE_PV:
            self.setCentralWidget(QLabel(
                "3D viewport unavailable — install pyvista pyvistaqt vtk."))
            return

        # central digital-twin viewport (its own instance)
        self.viewport = RobotViewport(self.kin)
        self.setCentralWidget(self.viewport)
        self.viewport.update_joints(self.q0)
        self.viewport.show_cad(self.cad, pickable=True)
        if not np.allclose(self.T_base_cad, np.eye(4)):
            self.viewport.show_frame(self.T_base_cad)

        self._build_toolbar()
        self._build_dock()
        self._status()

        # work plane defaults to a horizontal plane through the CAD centre
        self._init_work_plane()
        self._install_controller()
        self.refresh()

    # ---- top toolbar (draw tools) ----------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Tools")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        grp = QActionGroup(self)
        grp.setExclusive(True)
        self._act_add = self._tool_action(tb, grp, "✏ Add points", _TOOL_ADD, True)
        self._act_move = self._tool_action(tb, grp, "✥ Move point", _TOOL_MOVE, False)
        self._act_sel = self._tool_action(tb, grp, "☞ Select / Orbit", _TOOL_SELECT, False)
        tb.addSeparator()

        self._insert_chk = QCheckBox("Insert at selected")
        self._insert_chk.setToolTip("New points are inserted after the selected "
                                    "row instead of appended to the end.")
        tb.addWidget(self._insert_chk)
        tb.addSeparator()

        for label, slot, sc in (
            ("Delete", self._delete_selected, "Del"),
            ("Clear", self._clear, None),
            ("Undo", self.undo, "Ctrl+Z"),
            ("Redo", self.redo, "Ctrl+Y"),
            ("Close loop", self._close_loop, None),
            ("Reverse", self._reverse, None),
        ):
            act = QAction(label, self)
            if sc:
                act.setShortcut(QKeySequence(sc))
            act.triggered.connect(slot)
            tb.addAction(act)

    def _tool_action(self, tb, grp, label, tool, checked) -> QAction:
        act = QAction(label, self)
        act.setCheckable(True)
        act.setChecked(checked)
        grp.addAction(act)
        act.triggered.connect(lambda _=False, t=tool: self._set_tool(t))
        tb.addAction(act)
        return act

    def _set_tool(self, tool: str) -> None:
        self._tool = tool
        self._set_hint()

    # ---- left dock (settings + list + export) ----------------------------
    def _build_dock(self) -> None:
        panel = QWidget()
        root = QVBoxLayout(panel)

        note = QLabel("All coordinates are relative to the robot base centre "
                      "(metres → shown in mm).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#4c566a;font-size:10px")
        root.addWidget(note)

        # --- placement ---
        place = QGroupBox("Point placement")
        pg = QGridLayout(place)
        self.snap_chk = QCheckBox("Snap to CAD surface")
        self.snap_chk.setChecked(self.cad is not None)
        self.snap_chk.setEnabled(self.cad is not None)   # inert without a CAD
        self.snap_chk.setToolTip("When on, clicks that hit the CAD snap onto it; "
                                 "misses fall to the work plane below."
                                 + ("" if self.cad is not None
                                    else "  (No CAD loaded — points land on the "
                                         "work plane.)"))
        pg.addWidget(self.snap_chk, 0, 0, 1, 2)
        self.normal_chk = QCheckBox("Use surface normal for orientation")
        self.normal_chk.setChecked(True)
        self.normal_chk.toggled.connect(self.refresh)
        pg.addWidget(self.normal_chk, 1, 0, 1, 2)

        pg.addWidget(QLabel("Work plane"), 2, 0)
        self.plane_frame = QComboBox()
        self.plane_frame.addItems(["Base XY (horizontal)", "CAD XY"])
        self.plane_frame.currentIndexChanged.connect(self._update_work_plane)
        pg.addWidget(self.plane_frame, 2, 1)
        pg.addWidget(QLabel("Plane height (mm)"), 3, 0)
        self.plane_h = QDoubleSpinBox()
        self.plane_h.setRange(-2000, 2000)
        self.plane_h.setDecimals(1)
        self.plane_h.valueChanged.connect(self._update_work_plane)
        pg.addWidget(self.plane_h, 3, 1)
        root.addWidget(place)

        # --- motion ---
        motion = QGroupBox("Motion / tool")
        mg = QGridLayout(motion)
        mg.addWidget(QLabel("Default move"), 0, 0)
        self.move_box = QComboBox()
        self.move_box.addItems(list(_MOVE_TYPES.keys()))
        mg.addWidget(self.move_box, 0, 1)
        mg.addWidget(QLabel("Approach"), 1, 0)
        self.approach_box = QComboBox()
        self.approach_box.addItems(list(_APPROACH.keys()))
        self.approach_box.currentIndexChanged.connect(self.refresh)
        mg.addWidget(self.approach_box, 1, 1)
        mg.addWidget(QLabel("Speed (m/s)"), 2, 0)
        self.speed = QDoubleSpinBox(); self.speed.setRange(0.001, 3.0)
        self.speed.setDecimals(3); self.speed.setValue(0.1)
        mg.addWidget(self.speed, 2, 1)
        mg.addWidget(QLabel("Accel (m/s²)"), 3, 0)
        self.accel = QDoubleSpinBox(); self.accel.setRange(0.01, 10.0)
        self.accel.setDecimals(2); self.accel.setValue(0.5)
        mg.addWidget(self.accel, 3, 1)
        mg.addWidget(QLabel("Blend (mm)"), 4, 0)
        self.blend = QDoubleSpinBox(); self.blend.setRange(0.0, 200.0)
        self.blend.setValue(2.0)
        mg.addWidget(self.blend, 4, 1)
        mg.addWidget(QLabel("Tool standoff (mm)"), 5, 0)
        self.standoff = QDoubleSpinBox(); self.standoff.setRange(-200, 200)
        self.standoff.setValue(0.0)
        self.standoff.setToolTip("Lift the TCP this far off each point along "
                                 "the approach axis (positive = away from part).")
        self.standoff.valueChanged.connect(self.refresh)
        mg.addWidget(self.standoff, 5, 1)
        mg.addWidget(QLabel("Lead-in (mm)"), 6, 0)
        self.lead_in = QDoubleSpinBox(); self.lead_in.setRange(0, 500)
        self.lead_in.setValue(0.0)
        self.lead_in.setToolTip("Add an approach move that starts this far "
                                "above the first point.")
        mg.addWidget(self.lead_in, 6, 1)
        mg.addWidget(QLabel("Lead-out (mm)"), 7, 0)
        self.lead_out = QDoubleSpinBox(); self.lead_out.setRange(0, 500)
        self.lead_out.setValue(0.0)
        mg.addWidget(self.lead_out, 7, 1)
        root.addWidget(motion)

        # --- waypoint list ---
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["#", "Move", "X", "Y", "Z (mm)", "OK"])
        for c, w in enumerate((28, 66, 52, 52, 58, 30)):
            self.tree.setColumnWidth(c, w)
        self.tree.currentItemChanged.connect(self._on_row_changed)
        self.tree.itemDoubleClicked.connect(self._edit_row_move_type)
        root.addWidget(self.tree, 1)

        # --- export ---
        exp = QGroupBox("Export")
        eg = QGridLayout(exp)
        b_script = QPushButton("Save .script…"); b_script.clicked.connect(self._save_script)
        b_prog = QPushButton("Add to Program"); b_prog.clicked.connect(self._add_to_program)
        b_clip = QPushButton("Copy URScript"); b_clip.clicked.connect(self._copy_urscript)
        b_close = QPushButton("Close"); b_close.clicked.connect(self.close)
        eg.addWidget(b_script, 0, 0); eg.addWidget(b_prog, 0, 1)
        eg.addWidget(b_clip, 1, 0); eg.addWidget(b_close, 1, 1)
        root.addWidget(exp)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        dock = QDockWidget("Tools & Settings", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(scroll)
        dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _status(self) -> None:
        self.statusBar().showMessage("")
        self._set_hint()

    def _set_hint(self) -> None:
        hints = {
            _TOOL_ADD: "Add: click to drop a waypoint (drag to orbit). "
                       "Ctrl/Shift+drag a joint to hand-guide.",
            _TOOL_MOVE: "Move: drag an existing waypoint to reposition it.",
            _TOOL_SELECT: "Select/Orbit: click a waypoint to select; drag to orbit.",
        }
        n = len(self.waypoints)
        self.statusBar().showMessage(f"{n} waypoint(s).   {hints.get(self._tool, '')}")

    # ---- work plane -------------------------------------------------------
    def _init_work_plane(self) -> None:
        z = 0.0
        try:
            if self.cad is not None and len(self.cad.vertices):
                z = float(self.cad.center()[2])
        except Exception:                          # noqa: BLE001
            pass
        self.plane_h.blockSignals(True)
        self.plane_h.setValue(z * 1000.0)
        self.plane_h.blockSignals(False)
        self._update_work_plane()

    def work_plane(self):
        """Return (point, unit-normal) of the current work plane, base frame."""
        return self._wp_point, self._wp_normal

    def _update_work_plane(self, *_):
        h = self.plane_h.value() / 1000.0
        if self.plane_frame.currentIndex() == 1 and \
                not np.allclose(self.T_base_cad, np.eye(4)):
            n = self.T_base_cad[:3, 2]
            base = self.T_base_cad[:3, 3]
        else:
            n = np.array([0.0, 0.0, 1.0])
            base = np.zeros(3)
        n = n / (np.linalg.norm(n) or 1.0)
        self._wp_normal = n
        self._wp_point = base + h * n
        size = self._scene_size()
        if _HAVE_PV:
            self.viewport.show_work_plane(self._wp_point, self._wp_normal, size=size)

    def _scene_size(self) -> float:
        reach = 1.0
        try:
            reach = float(np.sum(np.abs(self.kin.d)) + np.sum(np.abs(self.kin.a)))
        except Exception:                          # noqa: BLE001
            pass
        return max(1.0, reach * 2.0)

    # ---- controller -------------------------------------------------------
    def _install_controller(self) -> None:
        try:
            iren = self.viewport.interactor()
            if iren is not None:
                self._ctl = _PathDrawController(self, self.viewport, iren)
            else:
                self._ctl = None
        except Exception as exc:                   # noqa: BLE001
            log.warning("Path draw interaction unavailable: %s", exc)
            self._ctl = None

    # ---- tool state queried by the controller ----------------------------
    def current_tool(self) -> str:
        return self._tool

    def snap_enabled(self) -> bool:
        return self.snap_chk.isChecked()

    # ---- model mutations --------------------------------------------------
    def _snapshot(self) -> None:
        self._undo.append((copy.deepcopy(self.waypoints), self.selected))
        self._redo.clear()
        if len(self._undo) > 100:
            self._undo.pop(0)

    def add_waypoint(self, pos, normal, on_surface) -> None:
        # honour the snap toggle: if off, always place on the work plane
        if not self.snap_chk.isChecked():
            on_surface = False
            normal = None
        self._snapshot()
        wp = Waypoint(pos=np.asarray(pos, float),
                      move=self.move_box.currentText(),
                      on_surface=bool(on_surface),
                      normal=None if normal is None else np.asarray(normal, float))
        if self._insert_chk.isChecked() and 0 <= self.selected < len(self.waypoints):
            self.waypoints.insert(self.selected + 1, wp)
            self.selected += 1
        else:
            self.waypoints.append(wp)
            self.selected = len(self.waypoints) - 1
        self.refresh()

    def update_waypoint(self, idx, pos, normal, on_surface) -> None:
        if not (0 <= idx < len(self.waypoints)):
            return
        wp = self.waypoints[idx]
        wp.pos = np.asarray(pos, float)
        if self.snap_chk.isChecked():
            wp.on_surface = bool(on_surface)
            wp.normal = None if normal is None else np.asarray(normal, float)
        self.refresh(rebuild_rows=False)

    def commit_edit(self) -> None:
        # push an undo checkpoint after a drag completes
        self._undo.append((copy.deepcopy(self.waypoints), self.selected))
        self.refresh()

    def select_index(self, idx) -> None:
        self.selected = idx if 0 <= idx < len(self.waypoints) else -1
        self.refresh(rebuild_rows=False)

    def _delete_selected(self) -> None:
        if not (0 <= self.selected < len(self.waypoints)):
            return
        self._snapshot()
        self.waypoints.pop(self.selected)
        self.selected = min(self.selected, len(self.waypoints) - 1)
        self.refresh()

    def _clear(self) -> None:
        if not self.waypoints:
            return
        self._snapshot()
        self.waypoints.clear()
        self.selected = -1
        self.refresh()

    def _close_loop(self) -> None:
        if len(self.waypoints) < 3:
            return
        self._snapshot()
        first = self.waypoints[0]
        self.waypoints.append(Waypoint(pos=first.pos.copy(), move=first.move,
                                       on_surface=first.on_surface,
                                       normal=None if first.normal is None
                                       else first.normal.copy()))
        self.refresh()

    def _reverse(self) -> None:
        if len(self.waypoints) < 2:
            return
        self._snapshot()
        self.waypoints.reverse()
        self.refresh()

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append((copy.deepcopy(self.waypoints), self.selected))
        self.waypoints, self.selected = self._undo.pop()
        self.refresh()

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append((copy.deepcopy(self.waypoints), self.selected))
        self.waypoints, self.selected = self._redo.pop()
        self.refresh()

    def nearest_waypoint(self, x, y, threshold=16):
        """Index of the waypoint whose screen projection is nearest (x,y)."""
        if not self.waypoints or not _HAVE_PV:
            return None
        ren = self.viewport.plotter.renderer
        best, best_d = None, threshold ** 2
        for i, wp in enumerate(self.waypoints):
            ren.SetWorldPoint(float(wp.pos[0]), float(wp.pos[1]),
                              float(wp.pos[2]), 1.0)
            ren.WorldToDisplay()
            dx, dy, _ = ren.GetDisplayPoint()
            d = (dx - x) ** 2 + (dy - y) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    # ---- orientation / pose building -------------------------------------
    def _approach_vec(self) -> np.ndarray:
        return np.asarray(_APPROACH[self.approach_box.currentText()], float)

    def _build_poses(self):
        """Per-waypoint (poses (N,6), transforms [4x4]) in the base frame."""
        n = len(self.waypoints)
        if n == 0:
            return np.empty((0, 6)), []
        pts = np.array([w.pos for w in self.waypoints], float)
        g_appr = self._approach_vec()
        standoff = self.standoff.value() / 1000.0
        use_n = self.normal_chk.isChecked()
        poses, Ts = [], []
        for i, wp in enumerate(self.waypoints):
            if n == 1:
                tangent = np.array([1.0, 0.0, 0.0])
            elif i < n - 1:
                tangent = pts[i + 1] - pts[i]
            else:
                tangent = pts[i] - pts[i - 1]
            if use_n and wp.on_surface and wp.normal is not None:
                approach = -np.asarray(wp.normal, float)
            else:
                approach = g_appr
            R = ToolpathGenerator._orientation(tangent, approach)
            T = np.eye(4)
            T[:3, :3] = R
            # standoff lifts the TCP opposite the tool +Z (approach) axis
            T[:3, 3] = wp.pos - standoff * R[:, 2]
            Ts.append(T)
            poses.append(matrix_to_pose(T))
        return np.asarray(poses), Ts

    def _update_reachability(self, poses) -> None:
        for i, wp in enumerate(self.waypoints):
            try:
                res = self.kin.inverse(poses[i], q_init=self.q0)
                wp.reachable = bool(res.success)
            except Exception:                      # noqa: BLE001
                wp.reachable = True

    # ---- refresh (3D + list) ---------------------------------------------
    def refresh(self, rebuild_rows: bool = True) -> None:
        if not _HAVE_PV:
            return
        poses, Ts = self._build_poses()
        if len(poses):
            self._update_reachability(poses)
        colors = ["#bf616a" if not wp.reachable else "#88c0d0"
                  for wp in self.waypoints]
        pts = np.array([w.pos for w in self.waypoints], float) \
            if self.waypoints else None
        # show orientation triads on a sparse subset to avoid clutter
        show_T = None
        if Ts:
            step = max(1, len(Ts) // 14)
            show_T = [Ts[i] for i in range(0, len(Ts), step)]
        self.viewport.draw_waypoints(pts, selected=self.selected,
                                     orient_mats=show_T, colors=colors)
        if rebuild_rows:
            self._rebuild_rows()
        self._set_hint()

    def _rebuild_rows(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        for i, wp in enumerate(self.waypoints):
            mm = wp.pos * 1000.0
            item = QTreeWidgetItem([
                str(i + 1), wp.move,
                f"{mm[0]:.0f}", f"{mm[1]:.0f}", f"{mm[2]:.0f}",
                "✓" if wp.reachable else "✗",
            ])
            if not wp.reachable:
                for c in range(6):
                    item.setForeground(c, QBrush(QColor("#bf616a")))
            if i == self.selected:
                self.tree.setCurrentItem(item)
            self.tree.addTopLevelItem(item)
        self.tree.blockSignals(False)

    def _on_row_changed(self, cur, _prev) -> None:
        idx = self.tree.indexOfTopLevelItem(cur) if cur is not None else -1
        if idx != self.selected:
            self.selected = idx
            self.refresh(rebuild_rows=False)

    def _edit_row_move_type(self, item, _col) -> None:
        idx = self.tree.indexOfTopLevelItem(item)
        if not (0 <= idx < len(self.waypoints)):
            return
        keys = list(_MOVE_TYPES.keys())
        cur = self.waypoints[idx].move
        nxt = keys[(keys.index(cur) + 1) % len(keys)] if cur in keys else keys[0]
        self.waypoints[idx].move = nxt
        self.refresh()

    # ---- export -----------------------------------------------------------
    def _export_steps(self) -> List[ProgramStep]:
        """Compile waypoints (+ lead-in/out) into ordered ProgramSteps."""
        poses, Ts = self._build_poses()
        if not len(poses):
            return []
        sp, ac = self.speed.value(), self.accel.value()
        bl = self.blend.value() / 1000.0
        j_sp, j_ac = 1.05, 1.40
        steps: List[ProgramStep] = []
        seed = self.q0.copy()

        def lifted(T, dist):
            L = T.copy()
            L[:3, 3] = L[:3, 3] - dist * L[:3, 2]     # move opposite approach
            return matrix_to_pose(L)

        # lead-in: MoveJ down to a standoff above the first point
        if self.lead_in.value() > 0:
            pose = lifted(Ts[0], self.lead_in.value() / 1000.0)
            seed = self._append_move("MoveJ", pose, steps, seed, sp, ac, bl,
                                     j_sp, j_ac, name="approach")
        for i, wp in enumerate(self.waypoints):
            seed = self._append_move(wp.move, poses[i], steps, seed, sp, ac, bl,
                                     j_sp, j_ac)
        # lead-out: retract straight up off the last point
        if self.lead_out.value() > 0:
            pose = lifted(Ts[-1], self.lead_out.value() / 1000.0)
            seed = self._append_move("MoveL", pose, steps, seed, sp, ac, bl,
                                     j_sp, j_ac, name="retract")
        return steps

    def _append_move(self, move, pose, steps, seed, sp, ac, bl, j_sp, j_ac,
                     name=""):
        st = _MOVE_TYPES.get(move, StepType.MOVEL)
        res = None
        try:
            res = self.kin.inverse(pose, q_init=seed)
        except Exception:                          # noqa: BLE001
            res = None
        if st is StepType.MOVEJ:
            if res is not None and res.success:
                steps.append(ProgramStep(StepType.MOVEJ, name=name,
                                         q=res.q.tolist(), speed=j_sp, accel=j_ac))
                return res.q
            # IK failed → degrade to a linear move so nothing is silently lost
            st = StepType.MOVEL
        steps.append(ProgramStep(st, name=name, pose=list(map(float, pose)),
                                 speed=sp, accel=ac,
                                 blend=bl if st in (StepType.MOVEL,) else max(bl, 0.001)))
        return res.q if (res is not None and res.success) else seed

    def _build_program(self) -> Optional[Program]:
        steps = self._export_steps()
        if not steps:
            QMessageBox.information(self, "Path Editor",
                                   "Add at least one waypoint first.")
            return None
        prog = Program(name="CAD Path")
        prog.steps = steps
        return prog

    def _save_script(self) -> None:
        prog = self._build_program()
        if prog is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save URScript", "cad_path.script",
            "URScript (*.script);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(URScriptGenerator().generate(prog))
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {path}")

    def _copy_urscript(self) -> None:
        prog = self._build_program()
        if prog is None:
            return
        QApplication.clipboard().setText(URScriptGenerator().generate(prog))
        self.statusBar().showMessage("URScript copied to clipboard.")

    def _add_to_program(self) -> None:
        steps = self._export_steps()
        if not steps:
            QMessageBox.information(self, "Path Editor",
                                   "Add at least one waypoint first.")
            return
        if self.program_panel is None:
            QMessageBox.warning(self, "Path Editor",
                                "No Program Builder is connected.")
            return
        self.program_panel.add_program_steps(steps)
        self.statusBar().showMessage(
            f"Added {len(steps)} step(s) to the Program Builder.")

    # ---- teardown ---------------------------------------------------------
    def closeEvent(self, event):                   # noqa: N802 (Qt override)
        try:
            if _HAVE_PV and self.viewport is not None:
                self.viewport.close()
        except Exception:                          # noqa: BLE001
            pass
        super().closeEvent(event)
