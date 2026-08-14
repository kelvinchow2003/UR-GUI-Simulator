"""
cad/cad_importer.py
==================================================================
CAD loading, 3-point frame calibration, and toolpath generation.

Supported formats (best-effort, dependency-gated):
    * .stl / .obj / .ply          -> trimesh          (mesh)
    * .step / .stp / .iges / .igs -> pythonocc-core   (B-rep, tessellated)
    * .dxf                        -> ezdxf            (2D curves)

Everything degrades gracefully: if a backend is missing, that format is
reported as unavailable rather than crashing the app. Meshes are always
returned in a common :class:`LoadedCAD` container (vertices, faces, edges)
so the 3D viewport and toolpath tools never care where the geometry
originated.

Frame calibration
-----------------
The user picks three points on the imported model:

    origin  – sets the CAD frame position P
    x_point – defines the +X direction
    y_point – defines the XY plane (Y is orthogonalised, Z = X × Y)

From these we build ``T_base_cad`` (4x4). Any toolpath extracted in CAD
coordinates is then mapped to the robot base frame by

    T_base_tool = T_base_cad @ T_cad_path

which is exactly the composition requested in the spec.
==================================================================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---- optional backends, imported lazily -----------------------------------
try:
    import trimesh                       # type: ignore
    _HAVE_TRIMESH = True
except Exception:                        # noqa: BLE001
    _HAVE_TRIMESH = False

try:
    import ezdxf                         # type: ignore
    _HAVE_EZDXF = True
except Exception:                        # noqa: BLE001
    _HAVE_EZDXF = False

try:
    # pythonocc-core exposes a large API; we only need STEP/IGES readers.
    from OCC.Extend.DataExchange import read_step_file, read_iges_file  # type: ignore
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh              # type: ignore
    from OCC.Core.TopExp import TopExp_Explorer                         # type: ignore
    from OCC.Core.TopAbs import TopAbs_FACE                             # type: ignore
    from OCC.Core.BRep import BRep_Tool                                 # type: ignore
    from OCC.Core.TopLoc import TopLoc_Location                         # type: ignore
    _HAVE_OCC = True
except Exception:                        # noqa: BLE001
    _HAVE_OCC = False


SUPPORTED_MESH = {".stl", ".obj", ".ply", ".off", ".glb"}
SUPPORTED_STEP = {".step", ".stp", ".iges", ".igs"}
SUPPORTED_DXF = {".dxf"}


@dataclass
class LoadedCAD:
    """Common container for any imported geometry."""
    path: str
    vertices: np.ndarray                       # (N,3) float
    faces: np.ndarray = field(                  # (M,3) int, may be empty
        default_factory=lambda: np.empty((0, 3), int))
    edges: np.ndarray = field(                   # (K,2) int index pairs
        default_factory=lambda: np.empty((0, 2), int))
    polylines: List[np.ndarray] = field(default_factory=list)  # ordered curves
    units: str = "m"

    @property
    def is_mesh(self) -> bool:
        return self.faces.shape[0] > 0

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            z = np.zeros(3)
            return z, z
        return self.vertices.min(0), self.vertices.max(0)

    def center(self) -> np.ndarray:
        lo, hi = self.bounds
        return (lo + hi) / 2.0


class CADImportError(RuntimeError):
    pass


class CADImporter:
    """Loads CAD files into :class:`LoadedCAD`, honouring available backends."""

    @staticmethod
    def supported_formats() -> List[str]:
        fmts: List[str] = []
        if _HAVE_TRIMESH:
            fmts += sorted(SUPPORTED_MESH)
        if _HAVE_OCC:
            fmts += sorted(SUPPORTED_STEP)
        if _HAVE_EZDXF:
            fmts += sorted(SUPPORTED_DXF)
        return fmts

    def load(self, path: str, scale_to_m: float = 1.0) -> LoadedCAD:
        ext = os.path.splitext(path)[1].lower()
        if not os.path.exists(path):
            raise CADImportError(f"File not found: {path}")
        if ext in SUPPORTED_MESH:
            cad = self._load_mesh(path)
        elif ext in SUPPORTED_STEP:
            cad = self._load_step_iges(path, ext)
        elif ext in SUPPORTED_DXF:
            cad = self._load_dxf(path)
        else:
            raise CADImportError(f"Unsupported format '{ext}'.")
        if scale_to_m != 1.0:
            cad.vertices = cad.vertices * scale_to_m
            cad.polylines = [p * scale_to_m for p in cad.polylines]
        return cad

    # ---- backends ---------------------------------------------------------
    def _load_mesh(self, path: str) -> LoadedCAD:
        if not _HAVE_TRIMESH:
            raise CADImportError("trimesh not installed — cannot load meshes.")
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        v = np.asarray(mesh.vertices, float)
        f = np.asarray(mesh.faces, int)
        edges = np.asarray(mesh.edges_unique, int) if len(f) else np.empty((0, 2), int)
        return LoadedCAD(path=path, vertices=v, faces=f, edges=edges)

    def _load_step_iges(self, path: str, ext: str) -> LoadedCAD:
        if not _HAVE_OCC:
            raise CADImportError(
                "pythonocc-core not installed — STEP/IGES import unavailable. "
                "Install via conda (`conda install -c conda-forge pythonocc-core`), "
                "or export the model to STL."
            )
        shape = (read_step_file(path) if ext in (".step", ".stp")
                 else read_iges_file(path))
        BRepMesh_IncrementalMesh(shape, 0.5)     # tessellate B-rep
        verts: List[np.ndarray] = []
        faces: List[List[int]] = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)
            if tri is not None:
                trsf = loc.Transformation()
                base = len(verts)
                n = tri.NbNodes()
                for i in range(1, n + 1):
                    p = tri.Node(i).Transformed(trsf)
                    verts.append([p.X(), p.Y(), p.Z()])
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    faces.append([base + a - 1, base + b - 1, base + c - 1])
            exp.Next()
        v = np.asarray(verts, float) / 1000.0    # STEP is usually millimetres
        f = np.asarray(faces, int)
        return LoadedCAD(path=path, vertices=v, faces=f, units="mm->m")

    def _load_dxf(self, path: str) -> LoadedCAD:
        if not _HAVE_EZDXF:
            raise CADImportError("ezdxf not installed — cannot load DXF.")
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        polylines: List[np.ndarray] = []
        pts: List[np.ndarray] = []
        for e in msp:
            dxftype = e.dxftype()
            if dxftype == "LINE":
                s = np.array([e.dxf.start.x, e.dxf.start.y, e.dxf.start.z])
                t = np.array([e.dxf.end.x, e.dxf.end.y, e.dxf.end.z])
                polylines.append(np.vstack([s, t]))
                pts.extend([s, t])
            elif dxftype in ("LWPOLYLINE", "POLYLINE"):
                poly = np.array([[p[0], p[1], 0.0] for p in e.get_points("xy")]) \
                    if dxftype == "LWPOLYLINE" else \
                    np.array([[v.dxf.location.x, v.dxf.location.y, v.dxf.location.z]
                              for v in e.vertices])
                if len(poly):
                    polylines.append(poly)
                    pts.extend(list(poly))
            elif dxftype in ("CIRCLE", "ARC"):
                polylines.append(self._sample_arc(e, dxftype))
                pts.extend(list(polylines[-1]))
        v = (np.asarray(pts, float) if pts else np.empty((0, 3)))
        # DXF is typically in mm
        v = v / 1000.0
        polylines = [p / 1000.0 for p in polylines]
        return LoadedCAD(path=path, vertices=v, polylines=polylines, units="mm->m")

    @staticmethod
    def _sample_arc(e, dxftype: str, n: int = 48) -> np.ndarray:
        c = np.array([e.dxf.center.x, e.dxf.center.y, e.dxf.center.z])
        r = e.dxf.radius
        if dxftype == "CIRCLE":
            a0, a1 = 0.0, 2 * np.pi
        else:
            a0 = np.radians(e.dxf.start_angle)
            a1 = np.radians(e.dxf.end_angle)
            if a1 < a0:
                a1 += 2 * np.pi
        ang = np.linspace(a0, a1, n)
        return c + np.column_stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n)])


# ===========================================================================
#  Frame calibration
# ===========================================================================
class FrameCalibrator:
    """Builds T_base_cad from three picked points (origin, +X, XY-plane)."""

    @staticmethod
    def from_three_points(origin: Sequence[float],
                          x_point: Sequence[float],
                          y_point: Sequence[float]) -> np.ndarray:
        o = np.asarray(origin, float)
        px = np.asarray(x_point, float)
        py = np.asarray(y_point, float)

        x = px - o
        nx = np.linalg.norm(x)
        if nx < 1e-9:
            raise ValueError("X point coincides with origin.")
        x = x / nx

        y_raw = py - o
        # Gram-Schmidt: remove the X component to get an orthogonal Y
        y = y_raw - np.dot(y_raw, x) * x
        ny = np.linalg.norm(y)
        if ny < 1e-9:
            raise ValueError("Y point is colinear with the X axis.")
        y = y / ny

        z = np.cross(x, y)
        T = np.eye(4)
        T[:3, 0] = x
        T[:3, 1] = y
        T[:3, 2] = z
        T[:3, 3] = o
        return T

    @staticmethod
    def identity() -> np.ndarray:
        return np.eye(4)


# ===========================================================================
#  Toolpath generation
# ===========================================================================
@dataclass
class Toolpath:
    """A sequence of TCP poses in the robot base frame."""
    poses: np.ndarray                          # (N,6) [x,y,z,rx,ry,rz]
    source: str = "cad"

    def __len__(self) -> int:
        return len(self.poses)


class ToolpathGenerator:
    """
    Convert CAD features (edges, polylines, face boundaries) into robot
    toolpaths expressed in the base frame:  T_base_tool = T_base_cad · T_cad_path.

    The tool orientation is derived from a face/local normal so the TCP
    approaches the surface along -Z (typical for dispensing, routing,
    inspection). A constant approach vector can be forced instead.
    """

    def __init__(self, T_base_cad: Optional[np.ndarray] = None):
        self.T_base_cad = np.eye(4) if T_base_cad is None else np.asarray(T_base_cad, float)

    def set_frame(self, T_base_cad: np.ndarray) -> None:
        self.T_base_cad = np.asarray(T_base_cad, float)

    # ---- public API -------------------------------------------------------
    def from_polyline(self, points_cad: np.ndarray,
                      approach: Sequence[float] = (0, 0, -1),
                      offset_m: float = 0.0) -> Toolpath:
        """Map an ordered CAD polyline to base-frame TCP poses."""
        pts = np.asarray(points_cad, float).reshape(-1, 3)
        approach = np.asarray(approach, float)
        approach = approach / (np.linalg.norm(approach) or 1.0)
        poses = []
        for i, p in enumerate(pts):
            # tangent from finite difference (path X axis)
            if i < len(pts) - 1:
                tangent = pts[i + 1] - p
            else:
                tangent = p - pts[i - 1]
            R_cad = self._orientation(tangent, approach)
            T_cad_path = np.eye(4)
            T_cad_path[:3, :3] = R_cad
            T_cad_path[:3, 3] = p + offset_m * R_cad[:, 2]
            T_base_tool = self.T_base_cad @ T_cad_path
            poses.append(self._to_pose(T_base_tool))
        return Toolpath(poses=np.asarray(poses), source="polyline")

    def from_edges(self, cad: LoadedCAD, **kwargs) -> List[Toolpath]:
        """Generate one toolpath per polyline / per connected edge loop."""
        paths: List[Toolpath] = []
        if cad.polylines:
            for poly in cad.polylines:
                paths.append(self.from_polyline(poly, **kwargs))
            return paths
        # mesh: order boundary edges into loops
        for loop in self._edge_loops(cad):
            paths.append(self.from_polyline(loop, **kwargs))
        return paths

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _orientation(tangent: np.ndarray, approach: np.ndarray) -> np.ndarray:
        """Build a rotation whose Z = approach, X ≈ path tangent."""
        z = approach / (np.linalg.norm(approach) or 1.0)
        x = tangent - np.dot(tangent, z) * z
        nx = np.linalg.norm(x)
        if nx < 1e-9:
            # tangent parallel to approach — pick any orthogonal X
            x = np.array([1.0, 0, 0]) - z[0] * z
            nx = np.linalg.norm(x)
        x = x / nx
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        return R

    @staticmethod
    def _to_pose(T: np.ndarray) -> np.ndarray:
        # local import to avoid a hard dependency cycle at module load
        from robot.kinematics import matrix_to_pose
        return matrix_to_pose(T)

    def _edge_loops(self, cad: LoadedCAD) -> List[np.ndarray]:
        """Order mesh boundary edges into continuous point loops."""
        if cad.edges.shape[0] == 0:
            return []
        try:
            import networkx as nx
        except Exception:                        # noqa: BLE001
            # fall back to raw (unordered) edge midpoints
            return [cad.vertices[cad.edges[:, 0]]]
        # boundary edges appear once; approximate with all unique edges here
        g = nx.Graph()
        for a, b in cad.edges:
            g.add_edge(int(a), int(b))
        loops: List[np.ndarray] = []
        for comp in nx.connected_components(g):
            sub = g.subgraph(comp)
            try:
                order = list(nx.dfs_preorder_nodes(sub))
                loops.append(cad.vertices[order])
            except Exception:                    # noqa: BLE001
                continue
        return loops
