"""
robot/palletizer.py
==================================================================
Pallet definition, box-placement generation, and a full pick→place
palletizing **job planner** that:

    * lays boxes out on a pallet in a grid pattern, layer by layer;
    * plans a complete pick→place motion for every box (approach heights,
      grip, retract) as exportable :class:`robot.program.ProgramStep`s;
    * builds a dense joint trajectory for the digital twin, plus timeline
      *events* (reveal a placed box, attach/detach the carried box);
    * checks each box for reachability (IK) and collision against the
      static world + already-stacked boxes + the box in the gripper,
      reporting the first failure so the user learns whether the whole
      palletization is physically possible before running it for real.

Frames & conventions match the rest of the app: base-frame metres, TCP
pose ``[x,y,z,rx,ry,rz]`` (axis-angle). The tool approaches every pick and
place pointing straight **down** (tool +Z = world −Z).
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from robot.kinematics import (
    Kinematics, TrajectoryPlanner, IKResult, matrix_to_pose, matrix_to_rotvec,
)
from robot.collision import Box, CollisionWorld
from robot.posture import PostureOptimizer, PostureWeights
from robot.program import ProgramStep, StepType

# tool pointing straight down: tool +Z = world −Z (180° about X)
_R_DOWN = np.array([[1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0]])


def _rz(phi: float) -> np.ndarray:
    """3x3 rotation about +Z by phi (radians)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose_down(xyz) -> np.ndarray:
    """A downward-tool TCP pose at position xyz."""
    return np.concatenate([np.asarray(xyz, float), matrix_to_rotvec(_R_DOWN)])


# Supported stacking patterns (layer-level). "column" stacks every layer
# identically; "interlock" rotates alternate layers 90° so the vertical seams
# between boxes don't line up, which is what gives a real palletized load its
# stability; "brick" shifts alternate layers by half a box pitch for the same
# reason when the boxes are too square to benefit from rotation.
PATTERNS = ("column", "interlock", "brick")


# ===========================================================================
#  Pallet specification
# ===========================================================================
@dataclass
class PalletSpec:
    """A pallet plus the box grid to stack on it (all metres)."""
    name: str = "Pallet 1"
    size: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.6, 0.144]))
    T: np.ndarray = field(default_factory=lambda: _default_pose())
    box_size: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.15, 0.12]))
    box_weight_kg: float = 1.0     # per-box mass, for the payload safety check
    pattern: str = "column"        # column | interlock | brick  ("grid" == column)
    layers: int = 3
    box_gap: float = 0.005
    layer_gap: float = 0.0
    nx: int = 0                # 0 ⇒ auto-fit from pallet/box/gap
    ny: int = 0
    color: str = "#d0a24c"
    # Role in a run: "stack" = palletize onto it from the pick point/conveyor;
    # "source" = it starts full and the robot depalletizes boxes off it;
    # "destination" = the robot places depalletized boxes onto it.
    role: str = "stack"
    # Where the tool contacts each box, in the box's own frame (metres from the
    # box centre). NaN ⇒ auto = top-face centre. The user can set this so the
    # robot grips a box from the top, an edge, or an offset "touch point".
    grip_point: np.ndarray = field(
        default_factory=lambda: np.array([np.nan, np.nan, np.nan]))

    def __post_init__(self):
        self.size = np.asarray(self.size, float).reshape(3)
        self.box_size = np.asarray(self.box_size, float).reshape(3)
        self.T = np.asarray(self.T, float).reshape(4, 4)
        self.grip_point = np.asarray(self.grip_point, float).reshape(3)

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        # grip_point may hold NaN (⇒ "auto top-centre"); JSON has no NaN, so
        # store those components as null and restore them on load.
        gp = [None if np.isnan(v) else float(v) for v in self.grip_point]
        return {
            "name": self.name,
            "size": self.size.tolist(),
            "T": self.T.tolist(),
            "box_size": self.box_size.tolist(),
            "box_weight_kg": float(self.box_weight_kg),
            "pattern": self.pattern,
            "layers": int(self.layers),
            "box_gap": float(self.box_gap),
            "layer_gap": float(self.layer_gap),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "color": self.color,
            "role": self.role,
            "grip_point": gp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PalletSpec":
        gp = d.get("grip_point", [None, None, None])
        grip = np.array([np.nan if v is None else float(v) for v in gp], float)
        return cls(
            name=d.get("name", "Pallet 1"),
            size=np.asarray(d["size"], float),
            T=np.asarray(d["T"], float),
            box_size=np.asarray(d["box_size"], float),
            box_weight_kg=float(d.get("box_weight_kg", 1.0)),
            pattern=d.get("pattern", "column"),
            layers=int(d.get("layers", 3)),
            box_gap=float(d.get("box_gap", 0.005)),
            layer_gap=float(d.get("layer_gap", 0.0)),
            nx=int(d.get("nx", 0)),
            ny=int(d.get("ny", 0)),
            color=d.get("color", "#d0a24c"),
            role=d.get("role", "stack"),
            grip_point=grip,
        )

    def grip_point_local(self) -> np.ndarray:
        """Resolved box-local contact point (defaults to the top-face centre)."""
        gp = np.asarray(self.grip_point, float)
        if np.any(np.isnan(gp)):
            return np.array([0.0, 0.0, float(self.box_size[2]) / 2.0])
        return gp

    # ---- grid sizing ------------------------------------------------------
    def grid_counts(self) -> Tuple[int, int]:
        if self.nx > 0 and self.ny > 0:
            return self.nx, self.ny
        L, W, _ = self.size
        bl, bw, _ = self.box_size
        nx = int(np.floor((L + self.box_gap) / (bl + self.box_gap))) if bl > 0 else 0
        ny = int(np.floor((W + self.box_gap) / (bw + self.box_gap))) if bw > 0 else 0
        return max(nx, 0), max(ny, 0)

    def per_layer(self) -> int:
        """Boxes in the first (bottom) layer for the active pattern."""
        return len(_layer_cells(self, 0))

    def total_boxes(self) -> int:
        return sum(len(_layer_cells(self, k)) for k in range(max(self.layers, 0)))

    def pallet_box(self) -> Box:
        """The pallet slab itself, as a collision/visual box.

        ``self.T`` is the pallet's **floor-standing base** (bottom-face centre),
        so the slab is centred half its height above that point — i.e. it rests
        *on* the floor at ``T`` rather than being half-buried. Boxes generated by
        :func:`generate_placements` stack from the slab's top face, so with this
        convention a box's bottom sits exactly on the pallet.
        """
        centre = self.T[:3, 3] + self.T[:3, :3] @ np.array([0.0, 0.0, self.size[2] / 2.0])
        return Box.from_size_center(self.size, centre, name=f"{self.name}:slab",
                                    kind="pallet", R=self.T[:3, :3])


def _default_pose() -> np.ndarray:
    """Pallet centred 0.6 m in front (+X) of the base, sitting on the floor."""
    T = np.eye(4)
    T[:3, 3] = [0.6, 0.0, 0.0]
    return T


@dataclass
class BoxPlacement:
    T: np.ndarray                 # world centre transform
    half: np.ndarray              # half-extents
    layer: int = 0
    index: int = 0

    def to_box(self, name: str) -> Box:
        return Box(half=self.half, T=self.T, name=name, kind="box")


# ===========================================================================
#  Placement generation
# ===========================================================================
def _grid_cells(bl, bw, L, W, gap, phi) -> List[tuple]:
    """Row-major (x, y, yaw) box centres for a footprint (bl×bw) tiling an
    L×W pallet with a given gap, centred, all at yaw ``phi``."""
    nx = int(np.floor((L + gap) / (bl + gap))) if bl > 0 else 0
    ny = int(np.floor((W + gap) / (bw + gap))) if bw > 0 else 0
    if nx <= 0 or ny <= 0:
        return []
    span_x = nx * bl + (nx - 1) * gap
    span_y = ny * bw + (ny - 1) * gap
    x0 = -span_x / 2.0 + bl / 2.0
    y0 = -span_y / 2.0 + bw / 2.0
    return [(x0 + ix * (bl + gap), y0 + iy * (bw + gap), phi)
            for iy in range(ny) for ix in range(nx)]


def _layer_cells(spec: "PalletSpec", layer: int) -> List[tuple]:
    """(x, y, yaw) box centres in the pallet plane for one layer, per pattern.

    * column    — every layer identical (yaw 0).
    * interlock — alternate layers rotated 90° (boxes and grid), so seams don't
                  align vertically: the classic load-stabilising pattern.
    * brick     — alternate layers shifted half a box pitch in X (wrapping boxes
                  that fall off the edge back to the front), for square-ish boxes
                  that don't benefit from rotation.
    """
    bl, bw, _ = spec.box_size
    L, W, _ = spec.size
    gap = spec.box_gap
    pattern = (spec.pattern or "column").lower()
    odd = (layer % 2 == 1)

    if pattern == "interlock" and odd:
        return _grid_cells(bw, bl, L, W, gap, np.pi / 2.0)   # box rotated 90°
    cells = _grid_cells(bl, bw, L, W, gap, 0.0)
    if pattern == "brick" and odd and cells:
        pitch = bl + gap
        half_x = L / 2.0
        shifted = []
        for (x, y, phi) in cells:
            xs = x + pitch / 2.0
            if xs + bl / 2.0 > half_x + 1e-6:          # fell off the +X edge
                xs -= (spec.grid_counts()[0]) * pitch  # wrap to the front
            shifted.append((xs, y, phi))
        return shifted
    return cells


def generate_placements(spec: PalletSpec) -> List[BoxPlacement]:
    """Ordered box placements (bottom layer first, row-major) in the base frame.

    Each box carries its own orientation (yaw about the pallet normal) so the
    downstream planner can rotate the wrist to set rotated boxes — required for
    the interlock/brick stability patterns.
    """
    bl, bw, bh = spec.box_size
    L, W, H = spec.size
    half = spec.box_size / 2.0
    R_pallet = spec.T[:3, :3]
    placements: List[BoxPlacement] = []
    idx = 0
    for layer in range(max(spec.layers, 0)):
        z = H + bh / 2.0 + layer * (bh + spec.layer_gap)     # local z (on slab top)
        layer_boxes: List[np.ndarray] = []
        for (x, y, phi) in _layer_cells(spec, layer):
            local = np.array([x, y, z])
            T = np.eye(4)
            T[:3, :3] = R_pallet @ _rz(phi)                  # per-box yaw
            T[:3, 3] = R_pallet @ local + spec.T[:3, 3]
            layer_boxes.append(T)
        # Placement order within a layer: farthest-from-base first. The robot
        # then always builds *away* from itself and retreats toward the base, so
        # it never has to reach over an already-stacked box to set one behind it.
        # This ordering alone removes most self-collisions in a full stack; the
        # per-box motion planner in PalletJob handles whatever remains. Ties are
        # broken deterministically so the visual reveal order stays stable.
        layer_boxes.sort(key=lambda T: (-float(np.hypot(T[0, 3], T[1, 3])),
                                        -float(T[1, 3]), -float(T[0, 3])))
        for T in layer_boxes:
            placements.append(BoxPlacement(T=T, half=half.copy(),
                                           layer=layer, index=idx))
            idx += 1
    return placements


# ===========================================================================
#  Job options + report
# ===========================================================================
@dataclass
class JobOptions:
    pick_pose: np.ndarray = field(          # where a box is presented (box top)
        default_factory=lambda: _pose_down([-0.4, -0.4, 0.2]))
    pick_approach: float = 0.10             # lift above pick (m)
    place_approach: float = 0.12            # lift above place (m)
    speed_l: float = 0.25                   # m/s for MoveL
    accel_l: float = 1.2
    speed_j: float = 1.05                   # rad/s for MoveJ
    accel_j: float = 1.4
    margin: float = 0.02                    # collision safety margin (m)
    # Work-surface tolerance: how far the (deliberately over-inflated) arm
    # capsule may "penetrate" the pallet slab before it's flagged. The capsules
    # enclose the real, thinner links with slack, so a few cm of capsule overlap
    # with the work surface the tool must reach over is not a real crash. A true
    # plunge into the pallet penetrates far deeper and is still caught.
    surface_tol: float = 0.03
    gripper_do: int = 0                     # digital-out pin for the gripper
    # --- planning resolution / performance knobs ---
    sim_steps: int = 10                     # playback interpolation samples / leg
    coll_stride: int = 2                    # collision-check every Nth sample
    ik_max_iter: int = 80                   # IK iteration cap while planning
    ik_seeds: int = 3                       # posture seeds tried per target
    # Collision-aware posture optimisation: enumerate IK branches (elbow/wrist/
    # base variants) and pick the one that clears the world, instead of taking
    # whatever branch the seed happened to land on. Off ⇒ legacy seed-loop.
    smart_posture: bool = True

    @classmethod
    def fast(cls, **kw) -> "JobOptions":
        """A coarser preset for quick iteration (fewer samples/seeds/iters)."""
        base = dict(sim_steps=6, coll_stride=3, ik_max_iter=50, ik_seeds=2)
        base.update(kw)
        return cls(**base)


@dataclass
class BoxStatus:
    index: int
    layer: int
    reachable: bool = True
    collided: bool = False
    detail: str = ""


@dataclass
class PalletReport:
    ok: bool = True
    statuses: List[BoxStatus] = field(default_factory=list)
    first_failure: int = -1            # index of the first failing box
    first_fail_sample: int = -1        # sample index in sim_q_path of the failure
    message: str = ""


# ===========================================================================
#  The job planner
# ===========================================================================
class PalletJob:
    """Plan a full palletization for one :class:`PalletSpec`."""

    def __init__(self, kin: Kinematics, static_obstacles: List[Box],
                 spec: PalletSpec, opts: Optional[JobOptions] = None,
                 q_start: Optional[np.ndarray] = None,
                 radii: Optional[Dict[str, float]] = None):
        self.kin = kin
        self.static = list(static_obstacles)
        self.spec = spec
        self.opts = opts or JobOptions()
        self.radii = radii
        self.q_start = (np.asarray(q_start, float) if q_start is not None
                        else np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]))
        self.planner = TrajectoryPlanner(kin)
        # Collision-aware posture engine. Continuity is weighted high so the arm
        # stays on its current IK branch whenever that branch is already clear,
        # and only switches to a different branch when the near one would collide
        # (the collision-free bonus guarantees a clear branch always wins). This
        # keeps consecutive legs on a coherent posture rather than hopping.
        self.smart = bool(getattr(self.opts, "smart_posture", True))
        self.posture = PostureOptimizer(
            kin, PostureWeights(clearance=12.0, clearance_cap=0.08,
                                singularity=1.0, limit=1.0,
                                elbow_up=3.0, continuity=2.0))

    # ---- carried-box geometry (in the TCP frame) -------------------------
    def _carried_local(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The gripped box expressed in the TCP frame — constant for every box.

        Because the wrist yaws to match each box (see :meth:`_place_pose`), the
        box's pose *relative to the tool* is identical no matter how the box is
        rotated in its layer::

            R_tool = R_box · R_down   ⇒   R_box in tool frame = R_downᵀ
            tcp = box_centre + R_box·gp  ⇒  centre − tcp (tool frame) = −R_downᵀ·gp
        """
        half = self.spec.box_size / 2.0
        gp = self.spec.grip_point_local()            # box-local contact point
        local = np.eye(4)
        local[:3, :3] = _R_DOWN.T
        local[:3, 3] = -_R_DOWN.T @ gp
        return local, half

    @staticmethod
    def _grip_pose(place: BoxPlacement, grip_local: np.ndarray) -> np.ndarray:
        """TCP pose to grip a box (pick or place): tool at the box's grip point,
        pointing down but yawed to match the box's orientation."""
        contact = place.T[:3, 3] + place.T[:3, :3] @ np.asarray(grip_local, float)
        pose = np.eye(4)
        pose[:3, :3] = place.T[:3, :3] @ _R_DOWN     # down, yawed with the box
        pose[:3, 3] = contact
        return matrix_to_pose(pose)

    def _place_pose(self, place: BoxPlacement) -> np.ndarray:
        """TCP pose to set a box down on this job's pallet (see :meth:`_grip_pose`)."""
        return self._grip_pose(place, self.spec.grip_point_local())

    @staticmethod
    def _lift(pose: np.ndarray, dz: float) -> np.ndarray:
        p = np.asarray(pose, float).copy()
        p[2] += dz
        return p

    @staticmethod
    def _at_height(pose: np.ndarray, z: float) -> np.ndarray:
        p = np.asarray(pose, float).copy()
        p[2] = z
        return p

    def _clearance_height(self) -> float:
        """A safe transfer height above the whole finished stack, the pick, and
        any static solid the tool must traverse over — e.g. boxes already stacked
        on another pallet in a multi-pallet run. Routing the high traverse above
        those keeps the carried box from sweeping through them; a genuine clip is
        still caught by the collision check and reported."""
        s = self.spec
        stack_top = (float(s.T[2, 3]) + float(s.size[2])
                     + s.layers * float(s.box_size[2])
                     + max(s.layers - 1, 0) * s.layer_gap)
        pick_z = float(np.asarray(self.opts.pick_pose, float)[2])
        carry_h = float(s.box_size[2])                 # box hangs below the TCP
        static_top = max((float(b.T[2, 3] + b.half[2]) for b in self.static),
                         default=0.0)
        return max(stack_top, pick_z, static_top + carry_h) + self.opts.place_approach

    def _solve_best(self, pose, seed, world: Optional[CollisionWorld] = None,
                    margin: Optional[float] = None):
        """
        IK with posture selection. When ``smart_posture`` is on *and* a collision
        ``world`` is supplied, this defers to :class:`PostureOptimizer`, which
        enumerates the reachable IK branches and returns the one that clears the
        world (falling back to the least-bad posture if none do) — so the arm
        actively re-postures around an obstacle instead of taking whatever branch
        the seed landed on. High continuity weighting keeps it on the seed's
        branch whenever that branch is already collision-free.

        Otherwise it uses the legacy seed loop: try the chained seed plus a few
        canonical *elbow-up* seeds and keep the successful solution whose whole
        arm stays **highest** (largest minimum link-origin z), avoiding an
        elbow-down posture that dips under the table / into the pallet.
        """
        pose = np.asarray(pose, float)
        if self.smart and world is not None:
            m = self.opts.margin if margin is None else margin
            res = self.posture.solve(pose, world=world, q_ref=seed, margin=m,
                                     max_iter=int(self.opts.ik_max_iter))
            if res.best is not None:
                return IKResult(q=res.best.q, success=res.ok, iterations=0,
                                pos_error=0.0, rot_error=0.0)
            # nothing reachable — fall through to the legacy solver for a seed
            # it can at least report an error from

        theta = float(np.arctan2(pose[1], pose[0]))
        all_seeds = [
            np.asarray(seed, float),
            np.array([theta, -1.2, -1.6, -1.45, 1.5708, 0.0]),
            np.array([theta, -1.6, -1.2, -1.50, 1.5708, 0.0]),
            np.array([theta, -2.0, -1.3, -1.40, 1.5708, 0.0]),
        ]
        seeds = all_seeds[:max(1, int(self.opts.ik_seeds))]
        max_iter = int(self.opts.ik_max_iter)
        best = None
        best_score = -np.inf
        for s in seeds:
            res = self.kin.inverse(pose, q_init=s, max_iter=max_iter)
            if not res.success:
                if best is None:
                    best = res
                continue
            frames = self.kin.fk_frames(res.q)
            min_z = min(float(f[2, 3]) for f in frames[1:])   # exclude fixed base
            if min_z > best_score:
                best_score = min_z
                best = res
            if min_z > 0.05:            # already a cleanly elbow-up solution
                break
        return best

    def _carried_stack_overlap(self, q, carried_box: Box, placed: List[Box]):
        """Return a CollisionResult if the carried box *overlaps* (penetrates)
        any already-placed box at config ``q``, else None. Uses a tiny 2 mm
        tolerance so intended face-to-face contact between stacked boxes doesn't
        register — only a genuine crash does."""
        from robot.collision import box_box_distance, CollisionResult
        T_tcp = self.kin.fk_frames(np.asarray(q, float))[-1]
        cb = Box(half=carried_box.half, T=T_tcp @ carried_box.T,
                 name=carried_box.name, kind="box")
        for pb in placed:
            d = box_box_distance(cb, pb)
            if d < -0.002:                             # >2 mm penetration
                return CollisionResult(hit=True, link="carried_box",
                                       box=pb.name, distance=d)
        return None

    # ---- adaptive height & approach helpers ------------------------------
    def _adaptive_clear(self, placed: List[Box], place_pose: np.ndarray) -> float:
        """Lowest transfer height that still clears everything already on the
        pallet for THIS box — instead of one global height for the whole job.

        Early boxes (near-empty pallet) travel low and fast; the height only
        rises as the stack grows. The carried box hangs a full box-height below
        the TCP (grip at its top face), so the tool must ride at least that far
        above the tallest placed box. Never returns less than the global
        clearance would *need* — it's a floor-raiser, not a safety relaxation.
        """
        s = self.spec
        pallet_top = float(s.T[2, 3]) + float(s.size[2])
        placed_top = max((float(b.T[2, 3] + b.half[2]) for b in placed),
                         default=pallet_top)
        carry_h = float(s.box_size[2])
        pick_top = float(np.asarray(self.opts.pick_pose, float)[2])
        base = max(placed_top + carry_h, pick_top, float(place_pose[2]))
        return base + self.opts.place_approach

    def _open_side(self, place: BoxPlacement,
                   placed: List[Box]) -> Optional[np.ndarray]:
        """A world-frame XY unit vector pointing to the most open side of a
        placement — a side with no already-stacked neighbour — so the tool can
        tuck the box in diagonally from there instead of plunging straight down
        between two walls. Prefers the base-ward side (the arm naturally reaches
        from there). Returns None when the box is free on all sides, in which
        case a plain vertical descent is fine."""
        s = self.spec
        R = s.T[:3, :3]
        origin = s.T[:3, 3]
        px = float(s.box_size[0] + s.box_gap)
        py = float(s.box_size[1] + s.box_gap)
        tgt = R.T @ (place.T[:3, 3] - origin)            # target centre, local
        occ = [(R.T @ (b.T[:3, 3] - origin))[:2] for b in placed]

        def is_occ(cx, cy) -> bool:
            return any(abs(lx - cx) < px * 0.5 and abs(ly - cy) < py * 0.5
                       for lx, ly in occ)

        open_dirs = []
        for sx, sy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if not is_occ(tgt[0] + sx * px, tgt[1] + sy * py):
                w = R @ np.array([sx, sy, 0.0], float)
                n = float(np.linalg.norm(w[:2]))
                if n > 1e-9:
                    open_dirs.append(w[:2] / n)
        if len(open_dirs) == 4 or not open_dirs:
            return None                                  # free (or boxed) all round
        radial = place.T[:2, 3]
        radial = radial / (float(np.linalg.norm(radial)) + 1e-9)
        open_dirs.sort(key=lambda d: float(np.dot(d, radial)))   # base-ward first
        return open_dirs[0]

    def _plan_box(self, k: int, place: BoxPlacement, q_prev: np.ndarray,
                  placed: List[Box], q_pick_clear: np.ndarray, q_pick: np.ndarray,
                  world_ext: CollisionWorld, slab_world: CollisionWorld,
                  local_carry: np.ndarray, carry_half: np.ndarray,
                  clearance_bonus: float = 0.0,
                  side: Optional[np.ndarray] = None,
                  via_apex: bool = False,
                  pick_pose: Optional[np.ndarray] = None,
                  reveal_event: Optional[tuple] = None,
                  pick_event: Optional[tuple] = None) -> dict:
        """Plan ONE box's pick→place with a given avoidance strategy, build its
        local sample buffer, and collision-check it. The caller tries a schedule
        of strategies (higher clearance, side approach, mid-traverse apex) and
        commits the first collision-free result into the global timeline.

        Each leg is ``(move_type, pose, carrying, name, grip, event)`` so the
        route length can vary (an apex via-point adds a leg) while step export
        and timeline events stay index-independent. Returns a dict with the local
        samples/events, the export legs+qs, reachability, and the first collision
        (if any) as ``(result, local_sample_offset, leg_index)``.
        """
        o = self.opts
        stride = max(1, int(o.coll_stride))
        pick = np.asarray(o.pick_pose if pick_pose is None else pick_pose, float)
        pick_clear = self._at_height(pick, self._clearance_height())

        place_pose = self._place_pose(place)
        z_clear = self._adaptive_clear(placed, place_pose) + clearance_bonus
        place_clear = self._at_height(place_pose, z_clear)

        # Side approach: nudge the high pre-descent point toward the open side so
        # the tool comes in on a diagonal rather than straight down into a slot
        # flanked by taller neighbours. The set-down target is unchanged.
        if side is not None:
            approach_hi = place_clear.copy()
            approach_hi[:2] += side * float(np.min(self.spec.box_size[:2])) * 0.6
        else:
            approach_hi = place_clear

        # Arm-avoid world for posture selection: external solids, the boxes
        # already stacked, AND the pallet slab / work surfaces. The slab is
        # included so the optimiser rejects an elbow-down branch that dips a link
        # into the pallet (it may reach the box *on* the slab, but the arm itself
        # must stay above it) — which is exactly the elbow-up posture a palletiser
        # wants. The real set-down check still applies the surface tolerance.
        avoid = CollisionWorld(list(world_ext.boxes) + list(placed)
                               + list(slab_world.boxes), self.radii)

        res_plc = self._solve_best(approach_hi, q_prev, world=avoid)
        q_place_clear = res_plc.q
        res_pl = self._solve_best(place_pose, q_place_clear, world=avoid)
        q_place = res_pl.q
        reachable = res_plc.success and res_pl.success

        # legs: (move_type, pose, carrying, name, grip_after, event)
        legs = [
            ("J", pick_clear, False, "pick approach", "", ""),
            ("L", pick, False, "pick", "close", ""),
            ("L", pick_clear, True, "lift", "", "carry"),
            ("J", approach_hi, True, "place approach", "", ""),
            ("L", place_pose, True, "place", "open", ""),
            ("L", approach_hi, False, "retract", "", "drop_reveal"),
        ]
        qs = [q_pick_clear, q_pick, q_pick_clear, q_place_clear, q_place, q_place_clear]

        # Via-point insertion: a joint-space traverse between two high points can
        # still bow *down* in Cartesian Z (joint-linear ≠ tool-linear), so raising
        # the endpoints alone doesn't guarantee the carried box clears the stack
        # mid-swing. When enabled, force the route up through an explicit apex over
        # the midpoint so the tool is provably high across the whole traverse.
        if via_apex and reachable:
            # Apex sits directly ABOVE the place slot (same reachable posture
            # family as the just-solved place-approach), only higher — so the arm
            # rises over the slot before descending and the carried box is provably
            # clear of the stack through the descent. Placing it over the geometric
            # midpoint instead would risk the shoulder singularity near the base.
            apex = approach_hi.copy()
            apex[2] = max(float(pick_clear[2]), float(approach_hi[2])) + 0.08
            res_apex = self._solve_best(apex, q_place_clear, world=avoid)
            if res_apex.success:
                legs.insert(3, ("J", apex, True, "traverse apex", "", ""))
                qs.insert(3, res_apex.q)

        # The current pallet's own already-placed boxes are a solid the ARM must
        # not pass through — the tool tip legitimately grazes the box it's setting
        # (absorbed by the penetration tolerance), but a forearm/upper-arm plunging
        # through the stack is a real crash. (The carried box vs this stack is a
        # separate, intended-contact check at set-down.)
        stack_world = CollisionWorld(placed, self.radii) if placed else None

        lsim: List[np.ndarray] = []
        levents: Dict[int, tuple] = {}
        leg_hit = None
        for li, leg in enumerate(legs):
            _, _, carrying, _, _, event = leg
            q0 = q_prev if li == 0 else qs[li - 1]
            q1 = qs[li]
            seg = self.planner.joint_move(q0, q1, o.sim_steps)
            carried_box = Box(half=carry_half, T=local_carry, name=f"box{k}") \
                if carrying else None
            base_off = len(lsim)
            lsim.extend(seg[1:].tolist())
            if leg_hit is None:
                for j in range(1, len(seg), stride):
                    res = world_ext.check(self.kin, seg[j], margin=o.margin,
                                          carried=carried_box,
                                          carried_obstacles=world_ext.boxes)
                    if res.hit:
                        leg_hit = (res, base_off + j - 1, li)
                        break
                    res_s = slab_world.check(self.kin, seg[j],
                                             margin=-o.surface_tol)
                    if res_s.hit:
                        leg_hit = (res_s, base_off + j - 1, li)
                        break
                    # arm links vs the stack being built (tool grazing tolerated)
                    if stack_world is not None:
                        res_st = stack_world.check(self.kin, seg[j],
                                                   margin=-o.surface_tol)
                        if res_st.hit:
                            leg_hit = (res_st, base_off + j - 1, li)
                            break
                # carried box vs the already-placed stack — checked only at the
                # settled set-down leg so it stays O(boxes), not O(boxes²·samples).
                if (leg_hit is None and leg[3] == "place"
                        and placed and carried_box is not None):
                    hit = self._carried_stack_overlap(q1, carried_box, placed)
                    if hit is not None:
                        leg_hit = (hit, len(lsim) - 1, li)
            if event == "carry":
                levents[base_off] = ("carry", local_carry.copy(), carry_half.copy())
                if pick_event is not None:          # e.g. hide the source box now
                    levents[base_off + 1] = pick_event
            elif event == "drop_reveal":
                levents[base_off] = (reveal_event if reveal_event is not None
                                     else ("drop_reveal", k + 1))
        return {"reachable": reachable, "leg_hit": leg_hit, "lsim": lsim,
                "levents": levents, "legs": legs, "qs": qs}

    # ---- the plan ---------------------------------------------------------
    def plan(self) -> Tuple[List[ProgramStep], np.ndarray, Dict[int, tuple], PalletReport]:
        placements = generate_placements(self.spec)
        report = PalletReport(ok=True)
        # Nothing fits — usually the box is bigger than the pallet (or gaps too
        # large). Report it clearly instead of a misleading "all 0 boxes OK".
        if not placements:
            report.ok = False
            report.message = ("No boxes fit on the pallet — the box footprint is "
                              "larger than the pallet (or the gaps are too large). "
                              "Increase the pallet size or reduce the box/gaps.")
            return [], np.asarray([self.q_start], float), {}, report

        # Payload safety: a box heavier than the robot's rated payload can't be
        # lifted no matter how reachable the pose is. Checked up front and folded
        # into the final verdict.
        payload = float(getattr(self.kin.model, "payload_kg", 0.0) or 0.0)
        payload_ok = not (payload > 0 and self.spec.box_weight_kg > payload)
        payload_msg = ""
        if not payload_ok:
            payload_msg = (f"⚠ Box mass {self.spec.box_weight_kg:.1f} kg exceeds the "
                           f"{self.kin.model.name} rated payload {payload:.0f} kg "
                           f"(before gripper mass) — use a bigger robot or lighter "
                           f"boxes.")

        steps: List[ProgramStep] = []
        sim: List[np.ndarray] = [self.q_start.copy()]
        events: Dict[int, tuple] = {}
        local_carry, carry_half = self._carried_local()

        pick = np.asarray(self.opts.pick_pose, float)
        z_clear = self._clearance_height()
        pick_clear = self._at_height(pick, z_clear)     # high transfer height
        pallet_slab = self.spec.pallet_box()

        # Two collision worlds with different intent:
        #   * external solids (guarding, pedestals, other pallets) get the full
        #     user safety margin — the arm must stay well clear of these.
        #   * work surfaces — the pallet slab AND any conveyor — are approached
        #     by the tool on purpose (place onto the pallet, pick a box off the
        #     belt), so they are checked arm-only with a penetration tolerance
        #     instead of the full margin. A conveyor also must NOT be tested
        #     against the carried box, or lifting a box off the belt (contact
        #     distance ≈ 0) would read as a crash. A genuine deep plunge into
        #     either still penetrates past the tolerance and is caught.
        surfaces = [b for b in self.static if b.kind == "conveyor"]
        solids = [b for b in self.static if b.kind != "conveyor"]
        world_ext = CollisionWorld(solids, self.radii)
        slab_world = CollisionWorld([pallet_slab] + surfaces, self.radii)
        placed: List[Box] = []

        # --- pick side is IDENTICAL for every box, so solve it ONCE ----------
        # (its poses don't depend on k). This alone removes ~half of all IK
        # solves, the dominant planning cost. Posture is chosen to clear the
        # external solids (the pick approaches over an empty area, away from the
        # pallet stack, so the slab isn't an obstacle here).
        res_pc = self._solve_best(pick_clear, self.q_start, world=world_ext)
        q_pick_clear = res_pc.q
        res_pk = self._solve_best(pick, q_pick_clear, world=world_ext)
        q_pick = res_pk.q
        pick_reachable = res_pc.success and res_pk.success

        for k, place in enumerate(placements):
            st = BoxStatus(index=k, layer=place.layer)
            q_prev = sim[-1]                 # config the arm arrives in

            # Collision-aware planning: instead of committing one fixed motion and
            # merely *reporting* a crash, try a schedule of avoidance strategies —
            # cheapest first — and keep the first one that's collision-free. The
            # arm effectively "thinks": lift higher, or tuck in from the box's open
            # side, before giving up. A feasible box normally succeeds on the first
            # (nominal) try, so clean stacks stay fast.
            strategies = [dict(clearance_bonus=0.0, side=None)]
            # Once the job already has a first failure, its verdict is sealed and
            # the animation stops there anyway — so don't burn the full retry
            # schedule on every remaining box; the nominal route still emits a
            # complete exportable program.
            if report.first_failure < 0:
                strategies += [dict(clearance_bonus=0.06, side=None),
                               dict(clearance_bonus=0.12, side=None)]
                side = self._open_side(place, placed)
                if side is not None:
                    strategies += [dict(clearance_bonus=0.06, side=side),
                                   dict(clearance_bonus=0.12, side=side)]
                # last resort: force the traverse up through an explicit apex so
                # the carried box provably clears the stack across the whole swing.
                strategies.append(dict(clearance_bonus=0.12, side=None, via_apex=True))
                if side is not None:
                    strategies.append(dict(clearance_bonus=0.12, side=side, via_apex=True))

            best = None
            for params in strategies:
                best = self._plan_box(k, place, q_prev, placed,
                                      q_pick_clear, q_pick, world_ext, slab_world,
                                      local_carry, carry_half, **params)
                if not (pick_reachable and best["reachable"]):
                    break               # unreachable target — re-routing won't help
                if best["leg_hit"] is None:
                    break               # collision-free strategy found — take it

            # commit the chosen (best) strategy into the global timeline
            base = len(sim)             # first sim sample belonging to this box
            box_start = base
            sim.extend(best["lsim"])
            for off, ev in best["levents"].items():
                events[base + off] = ev
            self._emit_steps(steps, best["legs"], best["qs"])

            st.reachable = pick_reachable and best["reachable"]
            leg_hit = best["leg_hit"]
            if leg_hit is not None:
                st.collided = True
                res = leg_hit[0]
                st.detail = f"{res.link} ↔ {res.box} (d={res.distance*1000:.0f} mm)"

            report.statuses.append(st)
            if (not st.reachable or st.collided) and report.first_failure < 0:
                report.first_failure = k
                report.ok = False
                report.first_fail_sample = (base + leg_hit[1]
                                            if (st.collided and leg_hit) else box_start)
                report.message = self._failure_message(k, place, st)

            placed.append(place.to_box(f"{self.spec.name}:box{k}"))

        if report.ok:
            report.message = (
                f"All {len(placements)} boxes reachable and collision-free "
                f"({self.spec.layers} layers, {self.spec.pattern} pattern).")
        # Payload overrides geometry: a reachable path you can't lift isn't runnable.
        if not payload_ok:
            report.ok = False
            report.message = (payload_msg if report.first_failure < 0
                              else payload_msg + "  Also: " + report.message)
        return steps, np.asarray(sim, float), events, report

    def _failure_message(self, k: int, place: "BoxPlacement", st: "BoxStatus") -> str:
        """A plain-language, actionable reason the palletization stopped."""
        where = f"Box {k + 1} (layer {place.layer + 1})"
        if not st.reachable:
            dist_mm = float(np.linalg.norm(place.T[:2, 3])) * 1000.0
            try:
                reach_mm = float(self.kin.model.reach_mm)
            except Exception:                          # noqa: BLE001
                reach_mm = 0.0
            reach_txt = (f" — it sits ~{dist_mm:.0f} mm from the base "
                         f"(arm reach {reach_mm:.0f} mm)") if reach_mm else ""
            return (f"{where} is unreachable{reach_txt}. Move the pallet closer to "
                    f"the base (lower Pallet pos X), shrink the pallet, or reposition it.")
        return (f"{where} collides — {st.detail}. Move the pallet/obstacle, raise the "
                f"approach height, or reduce the safety margin.")

    # ---- URScript steps ---------------------------------------------------
    def _emit_steps(self, steps: List[ProgramStep], legs, qs) -> None:
        """Emit exportable program steps for one box's route. Iterates the leg
        metadata so a variable-length route (e.g. one with a traverse-apex via)
        exports correctly and the gripper open/close land on the right legs."""
        o = self.opts
        for (mt, pose, _carry, name, grip, _event), q in zip(legs, qs):
            if mt == "J":
                steps.append(ProgramStep(StepType.MOVEJ, name=name,
                                         q=list(map(float, q)),
                                         speed=o.speed_j, accel=o.accel_j))
            else:
                steps.append(ProgramStep(StepType.MOVEL, name=name,
                                         pose=list(map(float, pose)),
                                         speed=o.speed_l, accel=o.accel_l))
            if grip == "close":
                steps.append(ProgramStep(StepType.GRIPPER_CLOSE, name="grip"))
            elif grip == "open":
                steps.append(ProgramStep(StepType.GRIPPER_OPEN, name="release"))


# ===========================================================================
#  Depalletize → palletize transfer (pallet-to-pallet)
# ===========================================================================
class TransferJob(PalletJob):
    """Move a full pallet's load onto another pallet: pick each box off the
    **source** (top layer first) and place it onto the **destination** in the
    normal stacking order.

    Reuses :class:`PalletJob`'s per-box motion planner and collision-avoidance
    strategies. ``self.spec`` is the destination (so ``_place_pose`` / clearance
    logic target it); the source stack shrinks box-by-box, and its remaining
    boxes stay in the obstacle set so the arm never crashes into what's left.

    ``base_src`` / ``base_dst`` are global box-index offsets so the timeline can
    hide the right source box and reveal the right destination box during play.
    """

    def __init__(self, kin, static_obstacles, source_spec: PalletSpec,
                 dest_spec: PalletSpec, opts: Optional[JobOptions] = None,
                 q_start: Optional[np.ndarray] = None,
                 radii: Optional[Dict[str, float]] = None,
                 base_src: int = 0, base_dst: int = 0):
        super().__init__(kin, static_obstacles, dest_spec, opts, q_start, radii)
        self.source_spec = source_spec
        self.base_src = int(base_src)
        self.base_dst = int(base_dst)

    def plan(self) -> Tuple[List[ProgramStep], np.ndarray, Dict[int, tuple], PalletReport]:
        src_pls = generate_placements(self.source_spec)
        dst_pls = generate_placements(self.spec)
        report = PalletReport(ok=True)
        if not src_pls or not dst_pls:
            report.ok = False
            report.message = ("Transfer needs boxes on the source pallet and free "
                              "slots on the destination pallet — check both sizes.")
            return [], np.asarray([self.q_start], float), {}, report

        n_src = len(src_pls)
        count = min(n_src, len(dst_pls))
        steps: List[ProgramStep] = []
        sim: List[np.ndarray] = [self.q_start.copy()]
        events: Dict[int, tuple] = {}
        local_carry, carry_half = self._carried_local()
        dst_slab = self.spec.pallet_box()
        src_slab = self.source_spec.pallet_box()
        src_grip = self.source_spec.grip_point_local()
        base_solids = [b for b in self.static if b.kind != "conveyor"]
        base_surfaces = [b for b in self.static if b.kind == "conveyor"]
        placed_dst: List[Box] = []

        for i in range(count):
            src_box = src_pls[n_src - 1 - i]           # depalletize top-first
            dst_place = dst_pls[i]                      # palletize in stacking order
            pick_pose = self._grip_pose(src_box, src_grip)

            # Remaining source boxes (below/around the one being picked) are a
            # work surface the tool descends into to grip the top box: checked
            # arm-only with the penetration tolerance (like the pallet slab and
            # the destination stack) so the wrist grazing a neighbour is fine but
            # a link plunging through the stack is still caught. The destination
            # growing stack is handled inside _plan_box (via the ``placed`` arg).
            remaining = [src_pls[j].to_box(f"{self.source_spec.name}:box{j}")
                         for j in range(n_src - 1 - i)]
            world_ext = CollisionWorld(base_solids, self.radii)
            slab_world = CollisionWorld(base_surfaces + [dst_slab, src_slab] + remaining,
                                        self.radii)

            pick_clear = self._at_height(pick_pose, self._clearance_height())
            # Posture-avoid the external solids plus the source boxes still below
            # the one being lifted, so the arm re-postures around the shrinking
            # source stack instead of plunging a link through it.
            avoid_pick = CollisionWorld(list(world_ext.boxes) + remaining, self.radii)
            res_pc = self._solve_best(pick_clear, sim[-1], world=avoid_pick)
            res_pk = self._solve_best(pick_pose, res_pc.q, world=avoid_pick)
            q_pick_clear, q_pick = res_pc.q, res_pk.q
            pick_reachable = res_pc.success and res_pk.success

            strategies = [dict(clearance_bonus=0.0, side=None)]
            if report.first_failure < 0:
                strategies += [dict(clearance_bonus=0.06, side=None),
                               dict(clearance_bonus=0.12, side=None)]
                sd = self._open_side(dst_place, placed_dst)
                if sd is not None:
                    strategies += [dict(clearance_bonus=0.06, side=sd),
                                   dict(clearance_bonus=0.12, side=sd)]
                strategies.append(dict(clearance_bonus=0.12, side=None, via_apex=True))

            src_g = self.base_src + (n_src - 1 - i)
            dst_g = self.base_dst + i
            best = None
            for params in strategies:
                best = self._plan_box(
                    i, dst_place, sim[-1], placed_dst, q_pick_clear, q_pick,
                    world_ext, slab_world, local_carry, carry_half,
                    pick_pose=pick_pose,
                    reveal_event=("box_show", dst_g),
                    pick_event=("box_hide", src_g), **params)
                if not (pick_reachable and best["reachable"]):
                    break
                if best["leg_hit"] is None:
                    break

            base = len(sim)
            sim.extend(best["lsim"])
            for off, ev in best["levents"].items():
                events[base + off] = ev
            self._emit_steps(steps, best["legs"], best["qs"])

            st = BoxStatus(index=i, layer=dst_place.layer)
            st.reachable = pick_reachable and best["reachable"]
            leg_hit = best["leg_hit"]
            if leg_hit is not None:
                st.collided = True
                res = leg_hit[0]
                st.detail = f"{res.link} ↔ {res.box} (d={res.distance*1000:.0f} mm)"
            report.statuses.append(st)
            if (not st.reachable or st.collided) and report.first_failure < 0:
                report.first_failure = i
                report.ok = False
                report.first_fail_sample = (base + leg_hit[1]
                                            if (st.collided and leg_hit) else base)
                report.message = self._failure_message(i, dst_place, st)

            placed_dst.append(dst_place.to_box(f"{self.spec.name}:box{i}"))

        if report.ok:
            report.message = (
                f"Transferred {count} boxes: {self.source_spec.name} → "
                f"{self.spec.name} ({self.spec.layers} layers, "
                f"{self.spec.pattern} pattern).")
        return steps, np.asarray(sim, float), events, report
