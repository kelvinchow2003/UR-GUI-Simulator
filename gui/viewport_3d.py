"""
gui/viewport_3d.py
==================================================================
Interactive 3D viewport — the digital twin canvas.

Robot geometry
--------------
When Universal Robots' **official visual meshes** are present under
``assets/meshes/<model>/visual/*.stl`` (fetch them with
``scripts/fetch_ur_meshes.py``), the twin renders the *real* UR CAD.
Each link mesh is mounted at its exact URDF link frame multiplied by the
per-link ``mesh_offset`` from UR's ``visual_parameters.yaml`` — i.e.
``T_link = fk_frames[k] @ mesh_offset``. If the meshes aren't downloaded,
a clean procedural stick-model is drawn from the same frames as a
fallback so the app always works.

Flicker-free
------------
Actors are created **once** and merely re-posed each frame via
``SetUserMatrix`` — nothing is added/removed during animation, so the
twin never blinks.

Interactive joint dragging
--------------------------
Hold **Ctrl** (or **Shift**) and left-drag a link to rotate that joint —
a software hand-guide that streams the target to the robot bridge. Plain
left-drag orbits the camera (trackball), wheel zooms.
==================================================================
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt, Signal

try:
    import pyvista as pv                     # type: ignore
    from pyvistaqt import QtInteractor       # type: ignore
    import vtk                               # type: ignore
    _HAVE_PV = True
except Exception:                            # noqa: BLE001
    _HAVE_PV = False

from robot.kinematics import Kinematics, pose_to_matrix

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS = os.path.join(_ROOT, "assets")

# procedural fallback colours
_LINK_COLORS = ["#3b4252", "#4c566a", "#5e81ac", "#81a1c1", "#88c0d0", "#8fbcbb"]
_JOINT_COLOR = "#ebcb8b"
_JOINT_HILITE = "#d08770"
_MESH_COLOR = "#e5e9f0"        # UR off-white shell
_MESH_HILITE = "#88c0d0"
_AXIS_COLORS = ("#bf616a", "#a3be8c", "#5e81ac")
_DRAG_GAIN = 0.008             # rad per pixel

# mesh link order == fk_frames index 0..6
_MESH_ORDER = ["base", "shoulder", "upper_arm", "forearm",
               "wrist_1", "wrist_2", "wrist_3"]


# ===========================================================================
#  Drag controller: Ctrl/Shift + left-drag rotates a picked joint
#
#  Observers are attached to the *interactor* (not a custom interactor
#  style). This is the robust choice: pyvista owns/replaces the interactor
#  style, so a custom style's observers can silently stop firing — but
#  interactor observers persist. While a joint drag is active we swap the
#  live style for a do-nothing vtkInteractorStyleUser so the camera can't
#  move, then restore the trackball style on release.
# ===========================================================================
if _HAVE_PV:

    class _DragController:
        def __init__(self, viewport: "RobotViewport", iren):
            self._vp = viewport
            self._iren = iren
            self._null_style = vtk.vtkInteractorStyleUser()
            self._saved_style = None
            self._drag = False
            self._joint = 0
            self._x0 = 0
            self._a0 = 0.0
            # high priority so we run before the style's own observers
            iren.AddObserver("LeftButtonPressEvent", self._on_press, 10.0)
            iren.AddObserver("MouseMoveEvent", self._on_move, 10.0)
            iren.AddObserver("LeftButtonReleaseEvent", self._on_release, 10.0)

        def _picked_joint(self) -> Optional[int]:
            x, y = self._iren.GetEventPosition()
            picker = vtk.vtkPropPicker()
            picker.Pick(x, y, 0, self._vp.plotter.renderer)
            actor = picker.GetActor()
            if actor is None:
                return None
            return self._vp._actor_joint.get(actor.GetAddressAsString(""))

        def _on_press(self, obj, evt):
            if not (self._iren.GetControlKey() or self._iren.GetShiftKey()):
                return
            j = self._picked_joint()
            import logging
            logging.getLogger("ur_gui.window").info(
                "hand-guide press: picked joint = %s", j)
            if j is None:
                return
            self._drag = True
            self._joint = j
            self._x0 = self._iren.GetEventPosition()[0]
            self._a0 = float(self._vp._q[j])
            self._vp.dragging = True
            self._vp._highlight_joint(j)
            # freeze the camera for the duration of the drag
            self._saved_style = self._iren.GetInteractorStyle()
            self._iren.SetInteractorStyle(self._null_style)

        def _on_move(self, obj, evt):
            if self._drag:
                x = self._iren.GetEventPosition()[0]
                self._vp._drag_joint(self._joint, self._a0 + (x - self._x0) * _DRAG_GAIN)

        def _on_release(self, obj, evt):
            if self._drag:
                self._drag = False
                self._vp.dragging = False
                self._vp._highlight_joint(None)
                if self._saved_style is not None:
                    self._iren.SetInteractorStyle(self._saved_style)
                    try:                        # clear any lingering rotate state
                        self._saved_style.StopState()
                    except Exception:           # noqa: BLE001
                        pass
                self._vp._drag_commit()


# ===========================================================================
#  Viewport widget
# ===========================================================================
class RobotViewport(QWidget):
    joints_dragged = Signal(object)

    def __init__(self, kin: Optional[Kinematics] = None, bridge=None, parent=None):
        super().__init__(parent)
        self.kin = kin
        self.bridge = bridge
        self._q = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        self.dragging = False

        # persistent actors
        self._mesh_actors: List = []          # (actor, link_name) when using real CAD
        self._link_actors: List = []          # (actor, height) procedural
        self._joint_actors: List = []
        self._tcp_actors: List = []
        self._actor_joint: Dict[str, int] = {}
        self._using_meshes = False
        self._actors_built = False
        self._cad_actor = None
        self._path_actors: List = []
        self._frame_actors: List = []
        self._edit_actors: List = []          # Path Editor overlay (markers/line/triads)
        self._workplane_actor = None          # Path Editor free-space work plane
        self._scene_actors: List = []         # obstacles + pallet slabs
        self._scene_meta: List = []           # [(actor, ref)] for click-picking
        self._placed_actors: List = []        # pallet boxes (revealed as stacked)
        self._carried_actor = None            # box currently in the gripper
        self._carried_half = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAVE_PV:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter.interactor)
            self._setup_scene()
            if kin is not None:
                self.update_joints(self._q)
        else:
            self.plotter = None
            msg = QLabel(
                "3D viewport unavailable.\n\n"
                "Install:  pip install pyvista pyvistaqt vtk")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet("color:#888;font-size:13px")
            layout.addWidget(msg)

    # ---- scene setup ------------------------------------------------------
    def _setup_scene(self) -> None:
        self.plotter.set_background("#eceff4", top="#d8dee9")
        self.plotter.add_axes(interactive=False)
        try:
            self.plotter.enable_anti_aliasing("msaa")
        except Exception:                       # noqa: BLE001
            pass
        try:
            grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                            i_size=3, j_size=3, i_resolution=15, j_resolution=15)
            self.plotter.add_mesh(grid, style="wireframe", color="#cfd6e0",
                                  line_width=1, name="ground", pickable=False)
        except Exception:                       # noqa: BLE001
            pass
        self._add_frame_triad(np.eye(4), scale=0.15, name_prefix="world")
        try:
            self.plotter.add_text("Ctrl/Shift + drag a joint to hand-guide",
                                  position="lower_left", font_size=9,
                                  color="#4c566a", name="hint")
        except Exception:                       # noqa: BLE001
            pass
        self._reset_camera_to_robot()
        # attach the joint-drag controller to the interactor
        self._drag_ctl = None
        try:
            iren = self._vtk_interactor()
            if iren is not None:
                self._drag_ctl = _DragController(self, iren)
        except Exception as exc:                # noqa: BLE001
            import logging
            logging.getLogger("ur_gui.window").warning(
                "Joint-drag interaction unavailable: %s", exc)

    def _vtk_interactor(self):
        """Return the underlying vtkRenderWindowInteractor, across pyvista APIs."""
        iren = getattr(self.plotter, "iren", None)
        if iren is not None:
            vtk_iren = getattr(iren, "interactor", None)
            if vtk_iren is not None:
                return vtk_iren
        rw = getattr(self.plotter, "render_window", None)
        if rw is not None:
            return rw.GetInteractor()
        return None

    def _reset_camera_to_robot(self) -> None:
        # frame the robot nicely from an isometric 3/4 view
        try:
            reach = 1.0
            if self.kin is not None:
                reach = float(np.sum(np.abs(self.kin.d)) + np.sum(np.abs(self.kin.a))) or 1.0
            self.plotter.camera_position = [
                (reach * 1.6, -reach * 1.6, reach * 1.3),
                (0, 0, reach * 0.4),
                (0, 0, 1),
            ]
            self.plotter.reset_camera()
        except Exception:                       # noqa: BLE001
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()

    def set_kinematics(self, kin: Kinematics) -> None:
        self.kin = kin
        self._teardown_robot()
        self.update_joints(self._q)
        self._reset_camera_to_robot()

    def set_bridge(self, bridge) -> None:
        self.bridge = bridge

    # ---- actor lifecycle --------------------------------------------------
    def _teardown_robot(self) -> None:
        if not _HAVE_PV:
            return
        for tup in self._mesh_actors:
            self._safe_remove(tup[0])
        for tup in self._link_actors:
            self._safe_remove(tup[0])
        for a in self._joint_actors + self._tcp_actors:
            self._safe_remove(a)
        self._mesh_actors.clear(); self._link_actors.clear()
        self._joint_actors.clear(); self._tcp_actors.clear()
        self._actor_joint.clear()
        self._actors_built = False

    def _mesh_stl_paths(self) -> Optional[Dict[str, str]]:
        """Return {link: stl_path} if UR's real meshes exist for this model."""
        if self.kin is None:
            return None
        paths = {}
        for link in _MESH_ORDER:
            try:
                rel = self.kin.mesh_path(link).replace(".dae", ".stl")
            except Exception:                   # noqa: BLE001
                return None
            p = os.path.join(_ASSETS, rel)
            if not os.path.exists(p):
                return None
            paths[link] = p
        return paths

    def _build_robot_actors(self) -> None:
        meshes = self._mesh_stl_paths()
        if meshes:
            self._build_mesh_actors(meshes)
        else:
            self._build_procedural_actors()
        # TCP triad (both modes)
        for axis in range(3):
            direction = [0, 0, 0]; direction[axis] = 1
            arrow = pv.Arrow(start=(0, 0, 0), direction=direction, scale=0.10)
            self._tcp_actors.append(self.plotter.add_mesh(
                arrow, color=_AXIS_COLORS[axis], name=f"tcp{axis}",
                pickable=False, reset_camera=False))
        self._actors_built = True

    def _build_mesh_actors(self, meshes: Dict[str, str]) -> None:
        self._using_meshes = True
        for k, link in enumerate(_MESH_ORDER):
            try:
                poly = pv.read(meshes[link])
            except Exception:                   # noqa: BLE001
                continue
            actor = self.plotter.add_mesh(
                poly, color=_MESH_COLOR, smooth_shading=True,
                specular=0.25, specular_power=12,
                name=f"mesh_{link}", reset_camera=False)
            self._mesh_actors.append((actor, link))
            # mesh k moves with joint k-1 (base is fixed → map to joint 0)
            self._actor_joint[actor.GetAddressAsString("")] = max(0, k - 1)

    def _build_procedural_actors(self) -> None:
        self._using_meshes = False
        frames = self.kin.fk_frames(np.zeros(6))
        origins = [f[:3, 3] for f in frames]
        for i in range(6):
            h = float(np.linalg.norm(origins[i + 1] - origins[i]))
            h = h if h > 1e-4 else 0.02
            radius = 0.045 if i < 2 else 0.035
            cyl = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1),
                              radius=radius, height=h, resolution=28)
            actor = self.plotter.add_mesh(
                cyl, color=_LINK_COLORS[i % len(_LINK_COLORS)],
                smooth_shading=True, name=f"link{i}", reset_camera=False)
            self._link_actors.append((actor, h))
            self._actor_joint[actor.GetAddressAsString("")] = i
            jr = 0.055 if i < 2 else 0.045
            sph = pv.Sphere(radius=jr, center=(0, 0, 0))
            jactor = self.plotter.add_mesh(sph, color=_JOINT_COLOR,
                                           name=f"joint{i}", reset_camera=False)
            self._joint_actors.append(jactor)
            self._actor_joint[jactor.GetAddressAsString("")] = i

    # ---- per-frame re-pose ------------------------------------------------
    def update_joints(self, q) -> None:
        self._q = np.asarray(q, float)
        if not _HAVE_PV or self.kin is None:
            return
        if not self._actors_built:
            self._build_robot_actors()

        frames = self.kin.fk_frames(self._q)
        if self._using_meshes:
            for actor, link in self._mesh_actors:
                k = _MESH_ORDER.index(link)
                self._set_matrix(actor, frames[k] @ self.kin.mesh_offset(link))
        else:
            origins = [f[:3, 3] for f in frames]
            for i, (actor, h) in enumerate(self._link_actors):
                self._set_matrix(actor, self._seg_matrix(origins[i], origins[i + 1], h))
            for i, actor in enumerate(self._joint_actors):
                T = np.eye(4); T[:3, 3] = origins[i]
                self._set_matrix(actor, T)
        for actor in self._tcp_actors:
            self._set_matrix(actor, frames[-1])
        self.plotter.render()

    # ---- dragging ---------------------------------------------------------
    def _drag_joint(self, j: int, angle: float) -> None:
        lo, hi = self.kin.q_min[j], self.kin.q_max[j]
        self._q[j] = float(np.clip(angle, lo, hi))
        self.update_joints(self._q)
        if self.bridge is not None:
            self.bridge.servo_target(self._q.tolist())

    def _drag_commit(self) -> None:
        if self.bridge is not None:
            self.bridge.move_j(self._q.tolist(), speed=0.8)
        self.joints_dragged.emit(self._q.copy())

    def _highlight_joint(self, j: Optional[int]) -> None:
        if self._using_meshes:
            for actor, link in self._mesh_actors:
                k = _MESH_ORDER.index(link)
                hot = (max(0, k - 1) == j) and j is not None
                try:
                    actor.prop.color = _MESH_HILITE if hot else _MESH_COLOR
                except Exception:               # noqa: BLE001
                    pass
        else:
            for idx, actor in enumerate(self._joint_actors):
                try:
                    actor.prop.color = _JOINT_HILITE if idx == j else _JOINT_COLOR
                except Exception:               # noqa: BLE001
                    pass
        if _HAVE_PV:
            self.plotter.render()

    # ---- transform helpers ------------------------------------------------
    @staticmethod
    def _seg_matrix(p0: np.ndarray, p1: np.ndarray, h: float) -> np.ndarray:
        d = p1 - p0
        length = np.linalg.norm(d)
        M = np.eye(4)
        if length < 1e-9:
            M[:3, 3] = p0
            return M
        zc = d / length
        z = np.array([0., 0., 1.])
        v = np.cross(z, zc); s = np.linalg.norm(v); c = float(np.dot(z, zc))
        if s < 1e-9:
            R = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
        else:
            k = v / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
        M[:3, :3] = R
        M[:3, 3] = (p0 + p1) / 2
        return M

    @staticmethod
    def _set_matrix(actor, M: np.ndarray) -> None:
        vm = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vm.SetElement(i, j, float(M[i, j]))
        try:
            actor.SetUserMatrix(vm)
        except Exception:                       # noqa: BLE001
            actor.user_matrix = M

    def _safe_remove(self, actor) -> None:
        try:
            self.plotter.remove_actor(actor, render=False)
        except Exception:                       # noqa: BLE001
            pass

    # ---- interactor / picking accessors (used by the Path Editor) ---------
    def interactor(self):
        """Public wrapper over the underlying vtkRenderWindowInteractor."""
        return self._vtk_interactor()

    def cad_actor(self):
        """Return the current CAD actor (for cell-picking), or None."""
        return self._cad_actor

    # ---- CAD overlay ------------------------------------------------------
    def show_cad(self, cad, pickable: bool = False) -> None:
        if not _HAVE_PV:
            return
        if self._cad_actor is not None:
            self._safe_remove(self._cad_actor)
            self._cad_actor = None
        if cad is None:
            self.plotter.render(); return
        try:
            if cad.is_mesh:
                faces = np.hstack(
                    [np.full((len(cad.faces), 1), 3), cad.faces]).astype(np.int64)
                mesh = pv.PolyData(cad.vertices, faces)
                self._cad_actor = self.plotter.add_mesh(
                    mesh, color="#b48ead", opacity=0.55, smooth_shading=True,
                    name="cad", pickable=pickable, reset_camera=False)
            elif cad.polylines:
                blocks = pv.MultiBlock()
                for poly in cad.polylines:
                    blocks.append(pv.lines_from_points(poly))
                self._cad_actor = self.plotter.add_mesh(
                    blocks.combine(), color="#b48ead", line_width=2,
                    name="cad", pickable=pickable, reset_camera=False)
            self.plotter.reset_camera()
        except Exception:                       # noqa: BLE001
            pass
        self.plotter.render()

    # ---- calibration frame ------------------------------------------------
    def show_frame(self, T: Optional[np.ndarray], scale: float = 0.12) -> None:
        for a in self._frame_actors:
            self._safe_remove(a)
        self._frame_actors.clear()
        if _HAVE_PV and T is not None:
            self._add_frame_triad(T, scale=scale, name_prefix="cadframe",
                                  store=self._frame_actors)
            self.plotter.render()

    def _add_frame_triad(self, T, scale=0.12, name_prefix="f", store=None):
        if not _HAVE_PV:
            return
        o = T[:3, 3]
        for axis, color in zip(range(3), ("red", "green", "blue")):
            arrow = pv.Arrow(start=o, direction=T[:3, axis], scale=scale)
            a = self.plotter.add_mesh(arrow, color=color, pickable=False,
                                      name=f"{name_prefix}{axis}", reset_camera=False)
            if store is not None:
                store.append(a)

    # ---- toolpath overlay -------------------------------------------------
    def show_toolpath(self, poses: Optional[np.ndarray]) -> None:
        for a in self._path_actors:
            self._safe_remove(a)
        self._path_actors.clear()
        if not _HAVE_PV or poses is None or len(poses) == 0:
            if _HAVE_PV:
                self.plotter.render()
            return
        pts = np.asarray(poses)[:, :3]
        line = pv.lines_from_points(pts)
        self._path_actors.append(self.plotter.add_mesh(
            line, color="#d08770", line_width=4, name="path",
            pickable=False, reset_camera=False))
        self._path_actors.append(self.plotter.add_points(
            pts, color="#bf616a", point_size=6, name="path_pts",
            pickable=False, reset_camera=False))
        for i in range(0, len(poses), max(1, len(poses) // 12)):
            T = pose_to_matrix(poses[i])
            for axis, color in zip(range(3), ("red", "green", "blue")):
                arrow = pv.Arrow(start=T[:3, 3], direction=T[:3, axis], scale=0.03)
                self._path_actors.append(self.plotter.add_mesh(
                    arrow, color=color, name=f"pf{i}_{axis}",
                    pickable=False, reset_camera=False))
        self.plotter.render()

    # ---- Path Editor overlay ---------------------------------------------
    def draw_waypoints(self, pts, selected: int = -1, orient_mats=None,
                       colors=None) -> None:
        """
        Render an *editable* path preview: a sphere per waypoint, a polyline
        joining them, the selected point emphasised, and optional TCP
        orientation triads. Kept separate from :meth:`show_toolpath` so the
        two overlays never fight over actor names.
        """
        for a in self._edit_actors:
            self._safe_remove(a)
        self._edit_actors.clear()
        if not _HAVE_PV:
            return
        pts = np.asarray(pts, float).reshape(-1, 3) if pts is not None \
            else np.empty((0, 3))
        if len(pts) >= 2:
            line = pv.lines_from_points(pts)
            self._edit_actors.append(self.plotter.add_mesh(
                line, color="#5e81ac", line_width=3, name="edit_line",
                pickable=False, reset_camera=False))
        # per-point spheres (reachable=teal, unreachable=red, selected=orange)
        for i, p in enumerate(pts):
            if i == selected:
                col = "#d08770"
            elif colors is not None and i < len(colors):
                col = colors[i]
            else:
                col = "#88c0d0"
            r = 0.012 if i == selected else 0.009
            sph = pv.Sphere(radius=r, center=p)
            self._edit_actors.append(self.plotter.add_mesh(
                sph, color=col, name=f"edit_pt{i}", pickable=False,
                reset_camera=False))
            try:
                self._edit_actors.append(self.plotter.add_point_labels(
                    np.array([p]), [str(i + 1)], font_size=11,
                    text_color="#2e3440", show_points=False, shape=None,
                    name=f"edit_lbl{i}", reset_camera=False))
            except Exception:               # noqa: BLE001
                pass
        # orientation triads (small)
        if orient_mats is not None:
            for i, T in enumerate(orient_mats):
                o = np.asarray(T)[:3, 3]
                for axis, color in zip(range(3), ("red", "green", "blue")):
                    arrow = pv.Arrow(start=o, direction=np.asarray(T)[:3, axis],
                                     scale=0.03)
                    self._edit_actors.append(self.plotter.add_mesh(
                        arrow, color=color, name=f"edit_tri{i}_{axis}",
                        pickable=False, reset_camera=False))
        self.plotter.render()

    def show_work_plane(self, point, normal, size: float = 1.0) -> None:
        """Draw / update the translucent work plane free-space clicks land on."""
        if self._workplane_actor is not None:
            self._safe_remove(self._workplane_actor)
            self._workplane_actor = None
        if not _HAVE_PV or point is None:
            if _HAVE_PV:
                self.plotter.render()
            return
        try:
            plane = pv.Plane(center=np.asarray(point, float),
                             direction=np.asarray(normal, float),
                             i_size=size, j_size=size,
                             i_resolution=10, j_resolution=10)
            self._workplane_actor = self.plotter.add_mesh(
                plane, color="#a3be8c", opacity=0.12, name="workplane",
                pickable=False, reset_camera=False)
            # a wireframe edge so the plane reads clearly
            self.plotter.add_mesh(plane, style="wireframe", color="#a3be8c",
                                  opacity=0.35, line_width=1, name="workplane_wire",
                                  pickable=False, reset_camera=False)
        except Exception:                       # noqa: BLE001
            pass
        self.plotter.render()

    # ---- Scene: collision boxes / pallets / palletized boxes --------------
    def _add_box(self, T, half, color, opacity, name, visible=True):
        half = np.asarray(half, float)
        cube = pv.Cube(center=(0, 0, 0), x_length=2 * half[0],
                       y_length=2 * half[1], z_length=2 * half[2])
        actor = self.plotter.add_mesh(
            cube, color=color, opacity=opacity, name=name, pickable=False,
            reset_camera=False, show_edges=True, edge_color="#2e3440",
            line_width=1)
        self._set_matrix(actor, np.asarray(T, float))
        if not visible:
            try:
                actor.SetVisibility(False)
            except Exception:                   # noqa: BLE001
                pass
        return actor

    def set_scene_boxes(self, specs) -> None:
        """Render static obstacles + pallet slabs. ``specs``: dicts with
        T, half, color, opacity, name, optional ref + ``selected``. Rebuilt on
        each scene edit; a selected box is highlighted and its ref recorded so
        clicks in the 3D view map back to the scene item."""
        for a in self._scene_actors:
            self._safe_remove(a)
        self._scene_actors.clear()
        self._scene_meta.clear()
        if not _HAVE_PV:
            return
        for i, s in enumerate(specs or []):
            sel = bool(s.get("selected", False))
            opacity = s.get("opacity", 0.35)
            actor = self._add_box(
                s["T"], s["half"], s.get("color", "#bf616a"),
                min(0.85, opacity + 0.25) if sel else opacity, f"scene_{i}")
            if sel:
                try:
                    actor.prop.edge_color = "#ebcb8b"
                    actor.prop.line_width = 3
                except Exception:               # noqa: BLE001
                    pass
            self._scene_actors.append(actor)
            self._scene_meta.append((actor, s.get("ref")))
        self.plotter.render()

    def scene_pick(self, x, y):
        """Return the ref of the scene box under display coords (x, y), or None."""
        if not _HAVE_PV or not self._scene_meta:
            return None
        try:
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            picker.PickFromListOn()
            picker.InitializePickList()
            for a, _ref in self._scene_meta:
                picker.AddPickList(a)
            if picker.Pick(x, y, 0, self.plotter.renderer) and \
                    picker.GetActor() is not None:
                addr = picker.GetActor().GetAddressAsString("")
                for a, ref in self._scene_meta:
                    if a.GetAddressAsString("") == addr:
                        return ref
        except Exception:                       # noqa: BLE001
            return None
        return None

    def update_scene_actor(self, ref, T) -> None:
        """Re-pose a single scene box (used for smooth drag without a rebuild)."""
        if not _HAVE_PV:
            return
        for a, r in self._scene_meta:
            if r == ref:
                self._set_matrix(a, np.asarray(T, float))
                self.plotter.render()
                return

    def world_ray(self, x, y):
        """Two world points (near, far) along the click ray at display (x, y)."""
        ren = self.plotter.renderer
        ren.SetDisplayPoint(x, y, 0.0); ren.DisplayToWorld()
        w0 = np.array(ren.GetWorldPoint(), float)
        ren.SetDisplayPoint(x, y, 1.0); ren.DisplayToWorld()
        w1 = np.array(ren.GetWorldPoint(), float)
        p0 = w0[:3] / (w0[3] if abs(w0[3]) > 1e-12 else 1.0)
        p1 = w1[:3] / (w1[3] if abs(w1[3]) > 1e-12 else 1.0)
        return p0, p1

    def camera_horizontal_normal(self):
        """Unit horizontal view direction (for a vertical drag plane)."""
        try:
            d = np.asarray(self.plotter.camera.direction, float)
        except Exception:                       # noqa: BLE001
            d = np.array([1.0, 0.0, 0.0])
        d[2] = 0.0
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    def build_placed_boxes(self, placements, color="#c9a06a") -> None:
        """Create (hidden) actors for every pallet box, to reveal during play."""
        for a in self._placed_actors:
            self._safe_remove(a)
        self._placed_actors.clear()
        if not _HAVE_PV:
            return
        for i, pl in enumerate(placements or []):
            T = pl["T"] if isinstance(pl, dict) else pl.T
            half = pl["half"] if isinstance(pl, dict) else pl.half
            self._placed_actors.append(
                self._add_box(T, half, color, 0.92, f"placed_{i}", visible=False))
        self.plotter.render()

    def set_placed_visible(self, n) -> None:
        if not _HAVE_PV:
            return
        for i, a in enumerate(self._placed_actors):
            try:
                a.SetVisibility(i < n)
            except Exception:                   # noqa: BLE001
                pass
        self.plotter.render()

    def set_carried_box(self, T, half=None) -> None:
        """Re-pose (or create/remove) the box currently held at the TCP."""
        if not _HAVE_PV:
            return
        if T is None or half is None:
            if self._carried_actor is not None:
                self._safe_remove(self._carried_actor)
                self._carried_actor = None
                self._carried_half = None
                self.plotter.render()
            return
        half = np.asarray(half, float)
        if (self._carried_actor is None or self._carried_half is None
                or not np.allclose(half, self._carried_half)):
            if self._carried_actor is not None:
                self._safe_remove(self._carried_actor)
            self._carried_actor = self._add_box(np.eye(4), half, "#a3be8c",
                                                0.92, "carried")
            self._carried_half = half
        self._set_matrix(self._carried_actor, np.asarray(T, float))
        self.plotter.render()

    def clear_pallet_animation(self) -> None:
        self.set_carried_box(None)
        self.set_placed_visible(0)

    def reset_view(self) -> None:
        if _HAVE_PV:
            self._reset_camera_to_robot()

    def closeEvent(self, event):               # noqa: N802 (Qt override)
        if _HAVE_PV and self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:                   # noqa: BLE001
                pass
        super().closeEvent(event)
