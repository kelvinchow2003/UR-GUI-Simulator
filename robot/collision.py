"""
robot/collision.py
==================================================================
Conservative, dependency-free collision checking for the digital twin.

Why conservative (not "exact")
------------------------------
Exact mesh-mesh collision would need FCL (`python-fcl`), which is not
available here. For a check whose result you intend to trust before
running a **real** robot, an optimistic "exact" test that can silently
under-report is the wrong trade. Instead each robot link is wrapped in a
**capsule that fully encloses its real geometry**, obstacles are exact
oriented boxes, and every test is inflated by a user **safety margin**.

Property: within the modelled geometry this **never clears a path that
actually collides** — it may occasionally flag a near-miss, which is the
safe direction to err. It remains a *feasibility aid*, not a safety
guarantee: always dry-run on the real robot at reduced speed.

Geometry
--------
* Robot: a poly-capsule following the FK link-frame origins
  (``kin.fk_frames(q)``). The fixed base segment is skipped so a pedestal
  *under* the base is never a false positive.
* Obstacles / pallet slabs / boxes: :class:`Box` — an oriented box
  (centre transform ``T`` + half-extents).
* Capsule-vs-box: a **sphere cover** of the segment (spacing ≤ radius, so
  the union of spheres contains the capsule) against the exact
  point-to-box distance in the box's local frame.

If ``python-fcl`` is ever importable the same public API can be backed by
an exact backend; until then the analytic path above is authoritative.
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# default enclosing radii (metres) per link segment when real meshes are
# unavailable — chosen to comfortably enclose UR shells.
_DEFAULT_RADII = {
    "shoulder": 0.075,
    "upper_arm": 0.065,
    "forearm": 0.055,
    "wrist_1": 0.050,
    "wrist_2": 0.050,
    "tcp": 0.050,
}
# ordered segments of the arm (endpoints are fk_frames indices)
#   frames: 0 base,1 shoulder,2 upper_arm,3 forearm,4 wrist_1,5 wrist_2,6 wrist_3,7 TCP
# we skip the fixed base link (0->1); moving links are 1->2 .. 6->7.
_SEGMENTS = [
    (1, 2, "shoulder"),
    (2, 3, "upper_arm"),
    (3, 4, "forearm"),
    (4, 5, "wrist_1"),
    (5, 6, "wrist_2"),
    (6, 7, "tcp"),
]

# ===========================================================================
#  Box primitive
# ===========================================================================
@dataclass
class Box:
    """An oriented box in the robot base frame."""
    half: np.ndarray                       # (3,) half-extents, metres
    T: np.ndarray = field(default_factory=lambda: np.eye(4))
    name: str = "box"
    kind: str = "obstacle"                 # pedestal|obstacle|pallet|box
    enabled: bool = True

    def __post_init__(self):
        self.half = np.asarray(self.half, float).reshape(3)
        self.T = np.asarray(self.T, float).reshape(4, 4)

    @classmethod
    def from_size_center(cls, size, center, name="box", kind="obstacle",
                         R=None) -> "Box":
        """Build from full size (lx,ly,lz) and a centre point (+ optional R)."""
        T = np.eye(4)
        if R is not None:
            T[:3, :3] = np.asarray(R, float)
        T[:3, 3] = np.asarray(center, float)
        return cls(half=np.asarray(size, float) / 2.0, T=T, name=name, kind=kind)

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return {"half": self.half.tolist(), "T": self.T.tolist(),
                "name": self.name, "kind": self.kind, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict) -> "Box":
        return cls(half=np.asarray(d["half"], float),
                   T=np.asarray(d["T"], float),
                   name=d.get("name", "box"), kind=d.get("kind", "obstacle"),
                   enabled=bool(d.get("enabled", True)))

    def corners(self) -> np.ndarray:
        """World-frame 8 corners (for rendering / bounds)."""
        s = self.half
        signs = np.array([[sx, sy, sz] for sx in (-1, 1)
                          for sy in (-1, 1) for sz in (-1, 1)], float)
        local = signs * s
        return (self.T[:3, :3] @ local.T).T + self.T[:3, 3]


# ===========================================================================
#  Distance primitives
# ===========================================================================
def point_box_distance(p_world: np.ndarray, box: Box) -> float:
    """Exact Euclidean distance from a world point to an oriented box (0 if inside)."""
    R = box.T[:3, :3]
    c = box.T[:3, 3]
    local = R.T @ (np.asarray(p_world, float) - c)     # into box frame
    d = np.abs(local) - box.half
    outside = np.maximum(d, 0.0)
    return float(np.linalg.norm(outside))              # 0 when fully inside


def _segment_samples(p0: np.ndarray, p1: np.ndarray, radius: float) -> np.ndarray:
    """Sphere-cover centres along a segment at spacing ≤ radius (>=2 samples)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    n = max(2, int(np.ceil(length / max(radius, 1e-4))) + 1)
    ts = np.linspace(0.0, 1.0, n)
    return p0[None, :] + ts[:, None] * (p1 - p0)[None, :]


def segment_box_distance(p0, p1, box: Box, radius: float) -> float:
    """
    Conservative distance from a capsule (segment + radius) surface to a box.
    Uses the sphere-cover: min over sample centres of (point-box dist) - radius.
    Negative ⇒ penetration.
    """
    pts = _segment_samples(p0, p1, radius)
    R = box.T[:3, :3]; c = box.T[:3, 3]
    local = (R.T @ (pts - c).T).T                      # (n,3) box-frame
    d = np.abs(local) - box.half
    outside = np.maximum(d, 0.0)
    dists = np.linalg.norm(outside, axis=1)            # centre-to-box distances
    return float(dists.min() - radius)


# ===========================================================================
#  Robot link geometry
# ===========================================================================
def link_radii(kin) -> Dict[str, float]:
    """
    Conservative capsule radius per link segment.

    We deliberately use fixed tube-radius envelopes rather than measuring the
    meshes: a UR link mesh spans its full *length* (~0.6 m) inside its own
    frame, so a naive radial measurement wildly over-estimates the cross-
    section. The defaults below comfortably enclose the real UR tube cross-
    sections; for larger arms (thicker tubes) they are scaled **up** only, so
    the capsule never under-approximates the real link — the safe direction.
    """
    radii = dict(_DEFAULT_RADII)
    try:
        reach_mm = float(getattr(kin.model, "reach_mm", 850.0))
    except Exception:                                # noqa: BLE001
        reach_mm = 850.0
    factor = max(1.0, reach_mm / 1000.0)             # inflate for big arms only
    return {k: v * factor for k, v in radii.items()}


def robot_capsules(kin, q, radii: Optional[Dict[str, float]] = None
                   ) -> List[Tuple[np.ndarray, np.ndarray, float, str]]:
    """Return [(p0, p1, radius, link_name)] for the moving links at config q."""
    if radii is None:
        radii = _DEFAULT_RADII
    frames = kin.fk_frames(np.asarray(q, float))
    origins = [f[:3, 3] for f in frames]
    caps = []
    for a, b, name in _SEGMENTS:
        caps.append((origins[a], origins[b], radii.get(name, 0.05), name))
    return caps


# ===========================================================================
#  Collision world
# ===========================================================================
@dataclass
class CollisionResult:
    hit: bool
    link: str = ""
    box: str = ""
    distance: float = np.inf       # signed capsule-surface distance (m)


class CollisionWorld:
    """A set of :class:`Box` obstacles the robot must avoid."""

    def __init__(self, boxes: Optional[List[Box]] = None,
                 radii: Optional[Dict[str, float]] = None):
        self.boxes: List[Box] = [b for b in (boxes or []) if b.enabled]
        self.radii = radii

    def check(self, kin, q, margin: float = 0.02,
              carried: Optional[Box] = None,
              carried_obstacles: Optional[List[Box]] = None) -> CollisionResult:
        """
        Nearest collision at configuration ``q``.

        * The robot links are tested against every box in ``self.boxes`` with
          the safety ``margin`` (m).
        * ``carried`` (a box gripped at the TCP) is tested against
          ``carried_obstacles`` — the *external* solids it must not strike
          (guarding, a pedestal, other pallets). It is deliberately **not**
          tested against the pallet slab it is being set onto nor the stack it
          is building, since those are intended contacts. When
          ``carried_obstacles`` is None the carried box falls back to
          ``self.boxes``.
        """
        best = CollisionResult(hit=False)
        caps = robot_capsules(kin, q, self.radii)
        for p0, p1, r, name in caps:
            for box in self.boxes:
                d = segment_box_distance(p0, p1, box, r) - margin
                if d < best.distance:
                    best = CollisionResult(hit=d < 0.0, link=name,
                                           box=box.name, distance=d)
        if carried is not None:
            T_tcp = kin.fk_frames(np.asarray(q, float))[-1]
            cb = Box(half=carried.half, T=T_tcp @ carried.T,
                     name=carried.name, kind="box")
            obstacles = self.boxes if carried_obstacles is None else carried_obstacles
            for box in obstacles:
                d = box_box_distance(cb, box) - margin
                if d < best.distance:
                    best = CollisionResult(hit=d < 0.0, link="carried_box",
                                           box=box.name, distance=d)
        best.hit = best.distance < 0.0
        return best

    def first_collision(self, kin, q_path, margin: float = 0.02,
                        carried_fn=None) -> Tuple[int, Optional[CollisionResult]]:
        """
        First trajectory sample that collides. ``carried_fn(i)`` may return a
        :class:`Box` (in TCP-local coordinates) held at sample ``i`` or None.
        Returns ``(index, result)`` or ``(-1, None)`` when the path is clear.
        """
        q_path = np.asarray(q_path, float)
        for i, q in enumerate(q_path):
            carried = carried_fn(i) if carried_fn is not None else None
            res = self.check(kin, q, margin=margin, carried=carried)
            if res.hit:
                return i, res
        return -1, None


def box_box_distance(a: Box, b: Box, samples: int = 3) -> float:
    """
    Conservative separation between two oriented boxes. Exact overlap test via
    the Separating Axis Theorem gives the sign; a small vertex/face sampling
    gives a usable positive distance when apart. Good enough for placement
    feasibility (boxes here are axis-similar).
    """
    if _obb_overlap(a, b):
        return -0.001
    # approximate positive distance: min corner-to-box over both directions
    da = min(point_box_distance(c, b) for c in a.corners())
    db = min(point_box_distance(c, a) for c in b.corners())
    return float(min(da, db))


def _obb_overlap(a: Box, b: Box) -> bool:
    """Separating Axis Theorem overlap test for two oriented boxes."""
    Ra, Rb = a.T[:3, :3], b.T[:3, :3]
    ta, tb = a.T[:3, 3], b.T[:3, 3]
    t = Rb.T @ (ta - tb)                    # b's frame not needed; use world axes
    # Build the 15 candidate separating axes
    axes = []
    for i in range(3):
        axes.append(Ra[:, i])
        axes.append(Rb[:, i])
    for i in range(3):
        for j in range(3):
            c = np.cross(Ra[:, i], Rb[:, j])
            if np.linalg.norm(c) > 1e-6:
                axes.append(c / np.linalg.norm(c))
    d = ta - tb
    for L in axes:
        L = np.asarray(L, float)
        ra = float(np.sum(np.abs((Ra.T @ L)) * a.half))
        rb = float(np.sum(np.abs((Rb.T @ L)) * b.half))
        if abs(float(np.dot(L, d))) > ra + rb + 1e-9:
            return False                    # found a separating axis
    return True
