"""
robot/rail.py
==================================================================
A **linear actuator the robot itself is mounted on** — a 7th axis.

Two shapes of the same idea, chosen by :attr:`RailSpec.axis`:

    * ``"Z"`` — a **vertical lift column**: the arm rides up and down so it
      can build (or strip) a tall pallet without the top layers falling out
      of reach and the bottom layers folding the arm into itself.
    * ``"X"`` / ``"Y"`` — a **horizontal track**: the arm traverses along a
      line, so one robot can serve a pick station and pallets that sit
      further apart than its own reach.

Frames
------
:attr:`RailSpec.origin` is the world→base transform the robot has at rail
**position 0** — i.e. the mount point. A position ``s`` (metres, signed,
along the axis expressed in the rail's own frame) slides the base to
:meth:`RailSpec.base_pose_at`. Everything downstream already follows a
moving base: :class:`robot.kinematics.Kinematics` premultiplies every FK
frame by it, so the twin, the collision capsules and the world poses IK
solves against all travel with the carriage for free.

Travel is clamped to ``[travel_min, travel_max]`` — the physical stroke of
the actuator, which is exactly what :func:`recommend_rail` sizes for you
from a pallet's box size and layer count.

Only the *fixed* structure (column / beam) is a collision obstacle. The
carriage rides with the robot base, so treating it as a static solid would
be wrong; it is drawn but never collided.
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from robot.collision import Box

# axis unit vectors, expressed in the rail's own (origin) frame
AXES: Dict[str, np.ndarray] = {
    "Z": np.array([0.0, 0.0, 1.0]),
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
}
AXIS_LABELS: Dict[str, str] = {
    "Z": "Z — vertical lift column",
    "X": "X — horizontal track",
    "Y": "Y — horizontal track",
}

#: How the rail is used during a palletizing job.
#:   fixed     — it never moves; it is just an adjustable mount height/position.
#:   per_layer — one indexed position per pallet layer (the classic lift column:
#:               index up a notch, build that layer, index again). The pick has
#:               to stay reachable from the same position.
#:   per_move  — the rail also indexes between the pick and the place, so the
#:               pick station and the pallet can each have their own best
#:               position. Most capable, but adds two actuator moves per box.
MODES: Tuple[str, ...] = ("fixed", "per_layer", "per_move")

# Reach-utilisation band a target should fall in, as a fraction of the arm's
# rated reach. Below the floor the target is inside the shoulder's dead zone
# (near-singular, and the arm folds into itself); above the ceiling it is at
# full stretch with no posture freedom left for obstacle avoidance.
U_LO = 0.30
U_HI = 0.90
#: minimum horizontal offset from the base axis — a UR cannot work in the
#: narrow cylinder directly above/below its own shoulder.
R_XY_MIN = 0.18
# The band a target should ideally land in. U_LO/U_HI above are the hard edges
# of what is workable at all; these are where the arm has real posture freedom.
# Scoring against a *band* rather than simply minimising stretch matters most
# for a single target: "as close as possible" would park the base directly over
# the box, which is exactly the fold-up posture a UR cannot hold.
U_GOOD_LO = 0.45
U_GOOD_HI = 0.80


# ===========================================================================
#  Rail specification
# ===========================================================================
@dataclass
class RailSpec:
    """A linear axis the robot base is mounted on (all metres)."""
    name: str = "Rail"
    enabled: bool = False
    axis: str = "Z"                 # Z = vertical lift, X/Y = horizontal track
    travel_min: float = 0.0         # stroke limits along the axis, from origin
    travel_max: float = 0.8
    position: float = 0.0           # current commanded position
    mode: str = "per_layer"         # see MODES
    speed: float = 0.30             # m/s
    accel: float = 0.60             # m/s^2
    #: world→base transform at position 0 (the mount point)
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))
    # ---- structure (rendered; the fixed part is also a collision solid) ----
    rail_width: float = 0.16        # column / beam cross-section
    #: Which way a vertical column stands from the robot, in degrees around the
    #: rail's own Z (180° = directly behind at −X). A lift column is a solid the
    #: arm must not swing into, so it belongs on the side away from the work —
    #: see :meth:`face_away_from`. Ignored by horizontal tracks, whose beam runs
    #: underneath the carriage.
    structure_angle: float = 180.0
    #: Distance from the robot's centreline to the column centre. A real lift
    #: column cantilevers the arm clear on a bracket precisely so the column is
    #: not standing inside the arm's own swing circle; mounting it hard against
    #: the base would put the upper arm through it on every swing that way.
    standoff: float = 0.45
    carriage: np.ndarray = field(
        default_factory=lambda: np.array([0.30, 0.30, 0.10]))
    collide: bool = True            # include the fixed structure in collision
    show_structure: bool = True
    color: str = "#5e81ac"
    # ---- how a rail move is expressed in exported code --------------------
    #: analog-out channel carrying the position setpoint (-1 ⇒ none)
    ao_pin: int = -1
    #: full-scale of that analog channel, in metres (setpoint = s / ao_scale)
    ao_scale: float = 1.0
    #: digital-out that tells the drive to go (-1 ⇒ none)
    do_pin: int = -1
    #: digital-in the drive raises when it is in position (-1 ⇒ no handshake)
    di_pin: int = -1

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float).reshape(4, 4)
        self.carriage = np.asarray(self.carriage, float).reshape(3)
        if self.axis not in AXES:
            self.axis = "Z"
        if self.mode not in MODES:
            self.mode = "per_layer"
        if self.travel_max < self.travel_min:
            self.travel_min, self.travel_max = self.travel_max, self.travel_min
        self.position = self.clamp(self.position)

    # ---- geometry ---------------------------------------------------------
    def axis_local(self) -> np.ndarray:
        return AXES[self.axis]

    def direction(self) -> np.ndarray:
        """The travel direction as a world-frame unit vector."""
        return self.origin[:3, :3] @ self.axis_local()

    def travel(self) -> float:
        return float(self.travel_max - self.travel_min)

    def clamp(self, s: float) -> float:
        return float(np.clip(float(s), self.travel_min, self.travel_max))

    def base_pose_at(self, s: float) -> np.ndarray:
        """World→base transform with the carriage at position ``s``."""
        T = self.origin.copy()
        T[:3, 3] = T[:3, 3] + self.direction() * float(s)
        return T

    def base_pose(self) -> np.ndarray:
        return self.base_pose_at(self.position)

    def base_origin_at(self, s: float) -> np.ndarray:
        """Just the base's world position at ``s`` (cheap — no matrix copy)."""
        return self.origin[:3, 3] + self.direction() * float(s)

    def move_time(self, s0: float, s1: float) -> float:
        """Trapezoidal-profile duration of a move (s), for cycle-time estimates."""
        d = abs(float(s1) - float(s0))
        if d < 1e-9:
            return 0.0
        v, a = max(self.speed, 1e-6), max(self.accel, 1e-6)
        if d <= v * v / a:                       # never reaches cruise speed
            return 2.0 * float(np.sqrt(d / a))
        return d / v + v / a

    # ---- structure geometry ----------------------------------------------
    def _local_box(self, size, centre, name, kind, yaw: float = 0.0) -> Box:
        """A box given in the rail's own frame, lifted into the world.

        ``yaw`` spins the box about the rail's local Z, so the column and its
        mounting bracket can point in any direction without the caller having
        to build a rotation itself.
        """
        size = np.asarray(size, float)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = self.origin[:3, :3] @ Rz
        T[:3, 3] = self.origin[:3, 3] + self.origin[:3, :3] @ np.asarray(centre, float)
        return Box(half=size / 2.0, T=T, name=name, kind=kind)

    def structure_box(self) -> Box:
        """The **fixed** part of the actuator (column for Z, beam for X/Y).

        Sized to span the whole stroke with a little end padding, and offset
        clear of the robot base so it never occupies the arm's own mount. A
        vertical column stands to the side given by :attr:`structure_angle`; a
        horizontal track gets a bed running under the carriage down towards the
        floor, so an elevated track reads as a real gantry rather than a
        floating beam.
        """
        w = float(self.rail_width)
        pad = w * 0.5
        lo, hi = self.travel_min - pad, self.travel_max + pad
        span = max(hi - lo, w)
        mid = (lo + hi) / 2.0
        cz = float(self.carriage[2])
        if self.axis == "Z":
            # a column standing beside the robot, far enough out that the arm
            # swings past it; the carriage bracket reaches back to it
            off = self._column_offset()
            a = np.radians(float(self.structure_angle))
            return self._local_box(
                [w, w, span],
                [off * np.cos(a), off * np.sin(a), mid],
                f"{self.name}:column", "rail", yaw=a)
        # horizontal track: a bed under the carriage, filling down to the floor
        top = -cz
        height = max(float(self.origin[2, 3]) + top, w)
        centre_z = top - height / 2.0
        if self.axis == "X":
            return self._local_box([span, w, height], [mid, 0.0, centre_z],
                                   f"{self.name}:beam", "rail")
        return self._local_box([w, span, height], [0.0, mid, centre_z],
                               f"{self.name}:beam", "rail")

    def anchor_to(self, base_pose) -> None:
        """Re-anchor the actuator so its **current** position lands the base on
        ``base_pose``.

        Fitting a rail, or dragging a rail-mounted robot to a new spot, should
        move the hardware to the robot — not teleport the robot to the hardware.
        """
        T = np.asarray(base_pose, float).reshape(4, 4).copy()
        self.origin = T                        # adopt the orientation first…
        origin = T.copy()                      # …so direction() is correct here
        origin[:3, 3] = T[:3, 3] - self.direction() * float(self.position)
        self.origin = origin

    def face_away_from(self, points) -> None:
        """Stand a vertical column in the widest gap between the stations the
        arm has to serve.

        Simply pointing it *opposite the average* of the work is wrong whenever
        the stations straddle the robot: the mean of a pick behind-left and a
        pallet in front lands the column right in the arc the arm swings
        through. Taking the largest empty sector instead puts the structure
        where the arm has no reason to go.
        """
        pts = np.asarray(points, float).reshape(-1, 3)
        if self.axis != "Z" or pts.size == 0:
            return
        local = (self.origin[:3, :3].T @ (pts - self.origin[:3, 3]).T).T
        ang = sorted(float(np.arctan2(p[1], p[0])) for p in local
                     if float(np.hypot(p[0], p[1])) > 1e-6)
        if not ang:
            return
        if len(ang) == 1:
            self.structure_angle = float(np.degrees(ang[0] + np.pi))
            return
        # widest circular gap between consecutive served directions
        gaps = [(ang[i + 1] - ang[i], ang[i]) for i in range(len(ang) - 1)]
        gaps.append((ang[0] + 2 * np.pi - ang[-1], ang[-1]))
        width, start = max(gaps, key=lambda g: g[0])
        self.structure_angle = float(np.degrees(start + width / 2.0))

    def _column_offset(self) -> float:
        """Radial distance from the base axis to the column centre — never less
        than the carriage plate itself needs."""
        w = float(self.rail_width)
        return max(float(self.standoff),
                   float(self.carriage[0]) / 2.0 + w / 2.0)

    def carriage_box(self, s: Optional[float] = None) -> Box:
        """The moving carriage the robot base is bolted to.

        On a vertical column this is the cantilever bracket reaching from the
        column out to the robot; on a horizontal track it is just the plate the
        base sits on. Either way its top face is the mount plane (rail-local
        z = 0), so the arm stands *on* it.
        """
        s = self.position if s is None else s
        cx, cy, cz = (float(self.carriage[0]), float(self.carriage[1]),
                      float(self.carriage[2]))
        travel = self.axis_local() * float(s)
        if self.axis == "Z":
            a = np.radians(float(self.structure_angle))
            reach = self._column_offset() + float(self.rail_width) / 2.0
            mid = reach / 2.0
            centre = travel + np.array([mid * np.cos(a), mid * np.sin(a),
                                        -cz / 2.0])
            return self._local_box([reach, cy, cz], centre,
                                   f"{self.name}:carriage", "rail", yaw=a)
        return self._local_box([cx, cy, cz],
                               travel + np.array([0.0, 0.0, -cz / 2.0]),
                               f"{self.name}:carriage", "rail")

    def collision_boxes(self) -> List[Box]:
        """Fixed structure only — the carriage travels with the base, so it is
        never a static obstacle (and the arm's own mount would always 'hit' it)."""
        if not (self.enabled and self.collide):
            return []
        return [self.structure_box()]

    def render_specs(self, s: Optional[float] = None) -> List[dict]:
        """Viewport specs for the structure and the carriage.

        The refs are ``("rail", 0)`` (fixed structure) and ``("rail", 1)``
        (carriage). They are *not* selectable — the rail is positioned from the
        panel, not dragged in the view — but naming them lets the carriage be
        re-posed on its own while a job animates.
        """
        if not (self.enabled and self.show_structure):
            return []
        out = []
        for i, b in enumerate((self.structure_box(), self.carriage_box(s))):
            out.append(dict(T=b.T, half=b.half, name=b.name, kind="rail",
                            color=self.color, opacity=0.55 if i == 0 else 0.85,
                            ref=("rail", i)))
        return out

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name, "enabled": bool(self.enabled), "axis": self.axis,
            "travel_min": float(self.travel_min),
            "travel_max": float(self.travel_max),
            "position": float(self.position), "mode": self.mode,
            "speed": float(self.speed), "accel": float(self.accel),
            "origin": self.origin.tolist(),
            "rail_width": float(self.rail_width),
            "structure_angle": float(self.structure_angle),
            "standoff": float(self.standoff),
            "carriage": self.carriage.tolist(),
            "collide": bool(self.collide),
            "show_structure": bool(self.show_structure),
            "color": self.color,
            "ao_pin": int(self.ao_pin), "ao_scale": float(self.ao_scale),
            "do_pin": int(self.do_pin), "di_pin": int(self.di_pin),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RailSpec":
        return cls(
            name=d.get("name", "Rail"),
            enabled=bool(d.get("enabled", False)),
            axis=d.get("axis", "Z"),
            travel_min=float(d.get("travel_min", 0.0)),
            travel_max=float(d.get("travel_max", 0.8)),
            position=float(d.get("position", 0.0)),
            mode=d.get("mode", "per_layer"),
            speed=float(d.get("speed", 0.30)),
            accel=float(d.get("accel", 0.60)),
            origin=np.asarray(d.get("origin", np.eye(4)), float),
            rail_width=float(d.get("rail_width", 0.16)),
            structure_angle=float(d.get("structure_angle", 180.0)),
            standoff=float(d.get("standoff", 0.45)),
            carriage=np.asarray(d.get("carriage", [0.30, 0.30, 0.10]), float),
            collide=bool(d.get("collide", True)),
            show_structure=bool(d.get("show_structure", True)),
            color=d.get("color", "#5e81ac"),
            ao_pin=int(d.get("ao_pin", -1)),
            ao_scale=float(d.get("ao_scale", 1.0)),
            do_pin=int(d.get("do_pin", -1)),
            di_pin=int(d.get("di_pin", -1)),
        )


# ===========================================================================
#  Travel recommendation
# ===========================================================================
@dataclass
class LayerPlan:
    """Where the rail should stand while the robot builds one pallet layer."""
    layer: int
    position: float          # rail position (m along the axis)
    feasible: bool
    grip_z: float            # world height of that layer's grip plane
    utilisation: float       # worst-case reach use at that position (0..1)
    detail: str = ""
    #: In ``per_move`` mode the carriage takes a different position for each box
    #: in the layer, so the layer occupies a *range* of the stroke rather than a
    #: single notch. Equal to ``position`` in the other modes.
    pos_min: float = 0.0
    pos_max: float = 0.0

    def __post_init__(self):
        if self.pos_min == 0.0 and self.pos_max == 0.0:
            self.pos_min = self.pos_max = self.position


@dataclass
class RailRecommendation:
    ok: bool = True
    axis: str = "Z"
    mode: str = "per_layer"
    pick_position: float = 0.0
    layers: List[LayerPlan] = field(default_factory=list)
    travel_min: float = 0.0
    travel_max: float = 0.0
    within_limits: bool = True
    message: str = ""
    detail: List[str] = field(default_factory=list)

    @property
    def travel(self) -> float:
        return float(self.travel_max - self.travel_min)

    def positions(self) -> List[float]:
        return [lp.position for lp in self.layers]

    def layer_position(self, layer: int, default: float = 0.0) -> float:
        for lp in self.layers:
            if lp.layer == layer:
                return lp.position
        return default


def _grip_points(spec, layer: int) -> np.ndarray:
    """World points the tool must reach for one layer of a pallet — the grip
    point of every box in that layer (the same contact the planner targets)."""
    from robot.palletizer import generate_placements
    gp = spec.grip_point_local()
    pts = [pl.T[:3, 3] + pl.T[:3, :3] @ gp
           for pl in generate_placements(spec) if pl.layer == layer]
    return np.asarray(pts, float).reshape(-1, 3)


def _utilisation(points: np.ndarray, base: np.ndarray, reach: float):
    """``(band_cost, worst utilisation, ok)`` for targets seen from a base origin.

    ``ok`` is False when any target falls outside the workable band — too far
    (past the rated reach), too close (inside the shoulder dead zone), or too
    nearly on the base's own vertical axis for the arm to point at it.

    ``band_cost`` is how far the worst target sits outside the *comfortable*
    band (0 when everything is inside it). Candidates are ranked by it first
    and by raw stretch second, so a position that keeps every target in the
    arm's happy range always beats one that merely gets closest to something.
    """
    if points.size == 0:
        return 0.0, 0.0, True
    d = points - np.asarray(base, float)
    r = np.linalg.norm(d, axis=1)
    r_xy = np.linalg.norm(d[:, :2], axis=1)
    u = r / max(reach, 1e-6)
    ok = bool(np.all(u <= U_HI) and np.all(u >= U_LO)
              and np.all(r_xy >= R_XY_MIN))
    cost = float(np.max(np.maximum(U_GOOD_LO - u, u - U_GOOD_HI).clip(min=0.0)))
    return cost, float(np.max(u)), ok


def _reach_pose(kin, pose: np.ndarray) -> bool:
    """Can the arm hold this downward-tool pose?

    Answered with the **same** engine the palletizer plans with
    (:class:`robot.posture.PostureOptimizer`), which enumerates the reachable
    IK branches instead of relying on one seed. A plain damped-least-squares
    solve from the mid-configuration gives up on poses the planner then builds
    without complaint — and a recommendation that contradicts the simulation
    directly below it is worse than no recommendation.
    """
    from robot.posture import PostureOptimizer
    opt = getattr(_reach_pose, "_cache", None)
    if opt is None or opt.kin is not kin:
        opt = PostureOptimizer(kin)
        _reach_pose._cache = opt
    return bool(opt.solve(pose, max_iter=100).ok)


def _verify_ik(kin, rail: RailSpec, s: float, points: np.ndarray,
               rot_down: Optional[np.ndarray] = None, samples: int = 4) -> bool:
    """Confirm with real IK that the arm can actually reach a layer from ``s``.

    Only the extreme targets are solved (farthest / nearest / the two lateral
    extremes): if the corners of a layer are reachable, the interior is. The
    base pose is set and restored around the check, so the caller's kinematics
    come back exactly as they were.
    """
    from robot.kinematics import matrix_to_rotvec
    if points.size == 0:
        return True
    R = rot_down if rot_down is not None else np.array([[1.0, 0.0, 0.0],
                                                        [0.0, -1.0, 0.0],
                                                        [0.0, 0.0, -1.0]])
    rv = matrix_to_rotvec(R)
    d = points - rail.base_origin_at(s)
    r = np.linalg.norm(d, axis=1)
    idx = {int(np.argmax(r)), int(np.argmin(r)),
           int(np.argmax(d[:, 1])), int(np.argmin(d[:, 1]))}
    idx = list(idx)[:max(1, int(samples))]

    saved = kin.base_pose()
    try:
        kin.set_base_pose(rail.base_pose_at(s))
        for i in idx:
            if not _reach_pose(kin, np.concatenate([points[i], rv])):
                return False
    finally:
        kin.set_base_pose(saved)
    return True


def _search_window(rail: RailSpec, spec, points_all: np.ndarray,
                   reach: float) -> Tuple[float, float]:
    """The span of rail positions worth evaluating, in rail coordinates.

    Deliberately wider than the configured stroke so the recommendation can say
    *"you need more travel than you have"* instead of silently clipping.
    """
    o = rail.origin[:3, 3]
    d = rail.direction()
    # project the work onto the axis: every target, plus a reach-sized margin
    if points_all.size:
        t = (points_all - o) @ d
        lo, hi = float(np.min(t)) - reach, float(np.max(t)) + 0.3 * reach
    else:
        lo, hi = -reach, reach
    lo = min(lo, rail.travel_min)
    hi = max(hi, rail.travel_max)
    if rail.axis == "Z":
        # a vertical column cannot drive the base below the floor
        floor_limit = -float(o[2]) / max(float(d[2]), 1e-6) if abs(d[2]) > 1e-6 else lo
        lo = max(lo, floor_limit)
    return float(lo), float(max(hi, lo + 1e-3))


def recommend_rail(kin, rail: RailSpec, spec, pick_pose=None,
                   mode: Optional[str] = None, samples: int = 41,
                   verify: bool = True,
                   limit_to_stroke: bool = False) -> RailRecommendation:
    """Work out where the rail should stand for each layer of ``spec`` — and
    therefore **how much travel the actuator needs**.

    For every layer the grip points of that layer's boxes are scored against a
    sweep of candidate rail positions. A position is workable when every target
    lands inside the arm's usable reach band (:data:`U_LO`…:data:`U_HI` of the
    rated reach, and clear of the shoulder dead zone); among the workable ones
    the least-stretched is chosen, then confirmed with real IK. In
    ``per_layer`` mode the fixed pick point must be reachable from the same
    position (the rail does not move mid-cycle); in ``per_move`` mode the pick
    gets its own position.

    The recommended stroke is the span of the chosen positions plus a small
    end margin. ``limit_to_stroke`` restricts the search to the rail's existing
    ``[travel_min, travel_max]`` — which is what the planner uses to index a
    job on the actuator the user actually has.
    """
    mode = mode or rail.mode
    reach = float(getattr(kin.model, "reach_mm", 1300.0)) / 1000.0
    n_layers = max(int(getattr(spec, "layers", 0)), 0)
    rec = RailRecommendation(axis=rail.axis, mode=mode)

    layer_pts = [_grip_points(spec, k) for k in range(n_layers)]
    layer_pts = [p for p in layer_pts if p.size]
    if not layer_pts:
        rec.ok = False
        rec.message = ("No boxes fit on this pallet, so there is nothing to size "
                       "the actuator against — check the pallet and box sizes.")
        return rec

    pick_pt = (np.asarray(pick_pose, float)[:3].reshape(1, 3)
               if pick_pose is not None else np.empty((0, 3)))
    all_pts = np.vstack(layer_pts + ([pick_pt] if pick_pt.size else []))

    if limit_to_stroke:
        lo, hi = rail.travel_min, rail.travel_max
    else:
        lo, hi = _search_window(rail, spec, all_pts, reach)
    cands = np.linspace(lo, hi, max(3, int(samples)))

    # the pick is fixed in the world; in per_layer mode it constrains every
    # layer, in per_move mode it only fixes the rail's own pick station.
    pick_couples = (mode != "per_move") and pick_pt.size > 0

    def best_for(points: np.ndarray, tag: str, do_verify: Optional[bool] = None):
        """Lowest-stretch workable candidate for a set of targets."""
        want_ik = verify if do_verify is None else bool(do_verify)
        scored = []
        for s in cands:
            cost, u, ok = _utilisation(points, rail.base_origin_at(float(s)),
                                       reach)
            if ok:
                scored.append((cost, u, float(s)))
        if not scored:
            # nothing works anywhere — report the least-bad so the user sees why
            fallback = min(
                ((_utilisation(points, rail.base_origin_at(float(s)), reach)[:2]
                  + (float(s),)) for s in cands), key=lambda t: (t[0], t[1]))
            return (fallback[2], fallback[1], False,
                    f"{tag}: out of reach at every rail position")
        scored.sort(key=lambda t: (t[0], t[1]))
        if want_ik:
            for _cost, u, s in scored[:5]:
                if _verify_ik(kin, rail, s, points):
                    return s, u, True, ""
            _cost, u, s = scored[0]
            return s, u, False, f"{tag}: geometrically in range but IK could not solve it"
        _cost, u, s = scored[0]
        return s, u, True, ""

    # ---- pick station -----------------------------------------------------
    if pick_pt.size and mode == "per_move":
        s_pick, _u, ok_pick, why = best_for(pick_pt, "pick point")
        rec.pick_position = s_pick
        if not ok_pick:
            rec.ok = False
            rec.detail.append(why)
    else:
        rec.pick_position = float(rail.position)

    # ---- a position per layer, or per box in per_move mode ---------------
    chosen: List[float] = []
    for k, pts in enumerate(layer_pts):
        grip_z = float(np.max(pts[:, 2]))
        if mode == "per_move":
            # Each set-down gets its own carriage position, so the layer is
            # sized by the spread of its boxes — that is what lets a horizontal
            # track serve a pallet longer than the arm's own reach. Only the two
            # extreme boxes are IK-verified; the rest lie between them.
            per = [best_for(pts[i:i + 1], f"layer {k + 1} box {i + 1}",
                            do_verify=False) for i in range(len(pts))]
            pos = [p[0] for p in per]
            ok = all(p[2] for p in per)
            why = next((p[3] for p in per if p[3]), "")
            lo_i, hi_i = int(np.argmin(pos)), int(np.argmax(pos))
            if ok and verify:
                for i in (lo_i, hi_i):
                    if not _verify_ik(kin, rail, pos[i], pts[i:i + 1]):
                        ok = False
                        why = (f"layer {k + 1}: geometrically in range but IK "
                               f"could not solve the end box")
                        break
            s = float(np.median(pos))
            u = max(per[lo_i][1], per[hi_i][1])
            lp = LayerPlan(layer=k, position=s, feasible=ok, grip_z=grip_z,
                           utilisation=u, detail=why,
                           pos_min=float(min(pos)), pos_max=float(max(pos)))
            chosen += [lp.pos_min, lp.pos_max]
        else:
            targets = np.vstack([pts, pick_pt]) if pick_couples else pts
            s, u, ok, why = best_for(targets, f"layer {k + 1}")
            lp = LayerPlan(layer=k, position=s, feasible=ok, grip_z=grip_z,
                           utilisation=u, detail=why)
            chosen.append(s)
        rec.layers.append(lp)
        if not ok:
            rec.ok = False
            if why:
                rec.detail.append(why)

    if mode == "per_move" and pick_pt.size:
        chosen.append(rec.pick_position)

    margin = 0.05                                   # 50 mm of end clearance
    rec.travel_min = float(min(chosen)) - margin
    rec.travel_max = float(max(chosen)) + margin
    rec.within_limits = (rec.travel_min >= rail.travel_min - 1e-6
                         and rec.travel_max <= rail.travel_max + 1e-6)
    rec.message = _describe(rec, rail, spec, mode)
    return rec


def _describe(rec: RailRecommendation, rail: RailSpec, spec, mode: str) -> str:
    """A plain-language summary of what the actuator has to do."""
    mm = 1000.0
    box_h = float(spec.box_size[2]) + float(spec.layer_gap)
    n = len(rec.layers)
    naive = max(n - 1, 0) * box_h                 # one box height per layer
    axis_word = "lift" if rec.axis == "Z" else f"travel along {rec.axis}"
    head = (f"{n} layers of {spec.box_size[0]*mm:.0f}×{spec.box_size[1]*mm:.0f}×"
            f"{spec.box_size[2]*mm:.0f} mm boxes ⇒ stack grows "
            f"{naive*mm:.0f} mm from layer 1 to layer {n}.")
    if not rec.ok:
        bad = [lp.layer + 1 for lp in rec.layers if not lp.feasible]
        why = ("  " + "  ".join(rec.detail)) if rec.detail else ""
        return (f"{head}  ✗ No workable rail position for layer(s) "
                f"{', '.join(map(str, bad))} — the pallet is out of reach at every "
                f"height. Move the pallet closer to the base or use a longer arm."
                f"{why}")
    span = rec.travel * mm
    fit = ("fits the current stroke" if rec.within_limits
           else f"NEEDS a longer stroke than the current "
                f"{rail.travel()*mm:.0f} mm")
    return (f"{head}  Recommended {axis_word}: "
            f"{rec.travel_min*mm:.0f} → {rec.travel_max*mm:.0f} mm "
            f"({span:.0f} mm of travel, {fit}).")


def layer_positions_for_job(kin, rail: RailSpec, spec, pick_pose=None,
                            samples: int = 21) -> Dict[int, float]:
    """The rail position to index to for each layer, **clamped to the stroke**
    the user actually configured — what :class:`robot.palletizer.PalletJob`
    drives the actuator with.

    Falls back to the rail's current position for any layer with no workable
    spot inside the stroke, so a job always plans (and then fails honestly in
    the reachability check) rather than raising.
    """
    if not rail.enabled or rail.mode == "fixed":
        return {}
    rec = recommend_rail(kin, rail, spec, pick_pose=pick_pose, mode=rail.mode,
                         samples=samples, verify=False, limit_to_stroke=True)
    out: Dict[int, float] = {}
    for lp in rec.layers:
        out[lp.layer] = rail.clamp(lp.position if lp.feasible else rail.position)
    return out
