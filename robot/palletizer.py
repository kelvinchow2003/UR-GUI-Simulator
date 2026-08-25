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
    Kinematics, TrajectoryPlanner, matrix_to_pose, matrix_to_rotvec,
)
from robot.collision import Box, CollisionWorld
from robot.program import ProgramStep, StepType

# tool pointing straight down: tool +Z = world −Z (180° about X)
_R_DOWN = np.array([[1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0]])


def _pose_down(xyz) -> np.ndarray:
    """A downward-tool TCP pose at position xyz."""
    return np.concatenate([np.asarray(xyz, float), matrix_to_rotvec(_R_DOWN)])


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
    pattern: str = "grid"
    layers: int = 3
    box_gap: float = 0.005
    layer_gap: float = 0.0
    nx: int = 0                # 0 ⇒ auto-fit from pallet/box/gap
    ny: int = 0
    color: str = "#d0a24c"
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
        nx, ny = self.grid_counts()
        return nx * ny

    def total_boxes(self) -> int:
        return self.per_layer() * max(self.layers, 0)

    def pallet_box(self) -> Box:
        """The pallet slab itself, as a collision/visual box."""
        return Box.from_size_center(self.size, self.T[:3, 3], name=f"{self.name}:slab",
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
def generate_placements(spec: PalletSpec) -> List[BoxPlacement]:
    """Ordered box placements (bottom layer first, row-major) in the base frame."""
    nx, ny = spec.grid_counts()
    if nx <= 0 or ny <= 0:
        return []
    bl, bw, bh = spec.box_size
    L, W, H = spec.size
    half = spec.box_size / 2.0
    # footprint the grid actually occupies, to centre it on the pallet
    span_x = nx * bl + (nx - 1) * spec.box_gap
    span_y = ny * bw + (ny - 1) * spec.box_gap
    x0 = -span_x / 2.0 + bl / 2.0
    y0 = -span_y / 2.0 + bw / 2.0
    placements: List[BoxPlacement] = []
    idx = 0
    for layer in range(spec.layers):
        z = H + bh / 2.0 + layer * (bh + spec.layer_gap)     # local z (on top of slab)
        for iy in range(ny):
            for ix in range(nx):
                local = np.array([x0 + ix * (bl + spec.box_gap),
                                  y0 + iy * (bw + spec.box_gap),
                                  z])
                T = np.eye(4)
                T[:3, :3] = spec.T[:3, :3]
                T[:3, 3] = spec.T[:3, :3] @ local + spec.T[:3, 3]
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
    gripper_do: int = 0                     # digital-out pin for the gripper
    sim_steps: int = 14                     # interpolation samples per segment


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

    # ---- carried-box geometry (in the TCP frame) -------------------------
    def _carried_local(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The gripped box expressed in the TCP frame — constant for every box
        since the grip point, box orientation and (downward) tool orientation
        are all constant. Derived so the touch point sits exactly at the TCP.
        """
        half = self.spec.box_size / 2.0
        gp = self.spec.grip_point_local()            # box-local contact point
        R_box = self.spec.T[:3, :3]
        # box centre relative to the TCP, in the (downward) tool frame:
        #   tcp_world = box_centre + R_box @ gp  ⇒  box_centre - tcp = -R_box @ gp
        t = _R_DOWN.T @ (-(R_box @ gp))
        local = np.eye(4)
        local[:3, :3] = _R_DOWN.T @ R_box
        local[:3, 3] = t
        return local, half

    def _place_pose(self, place: BoxPlacement) -> np.ndarray:
        """TCP pose to set a box down: tool at the box's grip point, pointing down."""
        gp = self.spec.grip_point_local()
        contact = place.T[:3, 3] + place.T[:3, :3] @ gp
        pose = np.eye(4); pose[:3, :3] = _R_DOWN; pose[:3, 3] = contact
        return matrix_to_pose(pose)

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
        """A safe transfer height above the whole finished stack + pick."""
        s = self.spec
        stack_top = (float(s.T[2, 3]) + float(s.size[2])
                     + s.layers * float(s.box_size[2])
                     + max(s.layers - 1, 0) * s.layer_gap)
        pick_z = float(np.asarray(self.opts.pick_pose, float)[2])
        return max(stack_top, pick_z) + self.opts.place_approach

    def _solve_best(self, pose, seed):
        """
        IK with posture selection: try the chained seed plus a few canonical
        *elbow-up* seeds (base rotated toward the target) and keep the
        successful solution whose whole arm stays **highest** (largest minimum
        link-origin z). This avoids the damped-least-squares solver settling
        into an elbow-down posture that dips under the table / into the pallet
        — a solution that is kinematically valid but not how a palletizer moves.
        """
        pose = np.asarray(pose, float)
        theta = float(np.arctan2(pose[1], pose[0]))
        seeds = [
            np.asarray(seed, float),
            np.array([theta, -1.2, -1.6, -1.45, 1.5708, 0.0]),
            np.array([theta, -1.6, -1.2, -1.50, 1.5708, 0.0]),
            np.array([theta, -2.0, -1.3, -1.40, 1.5708, 0.0]),
        ]
        best = None
        best_score = -np.inf
        for s in seeds:
            res = self.kin.inverse(pose, q_init=s)
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
        steps: List[ProgramStep] = []
        sim: List[np.ndarray] = [self.q_start.copy()]
        events: Dict[int, tuple] = {}
        local_carry, carry_half = self._carried_local()

        pick = np.asarray(self.opts.pick_pose, float)
        z_clear = self._clearance_height()
        pick_clear = self._at_height(pick, z_clear)     # high transfer height
        seed = self.q_start.copy()
        placed: List[Box] = []
        pallet_slab = self.spec.pallet_box()

        for k, place in enumerate(placements):
            st = BoxStatus(index=k, layer=place.layer)
            place_pose = self._place_pose(place)
            place_up = self._lift(place_pose, self.opts.place_approach)
            place_clear = self._at_height(place_pose, z_clear)

            # motion waypoints. Traverses (legs 0 & 3) happen at a common
            # clearance height above the whole stack, so the carried box never
            # sweeps through already-placed boxes during the joint-space move —
            # exactly how a real palletizer routes: up, across high, straight down.
            legs = [
                ("J", pick_clear, False),   # 0 go high above pick
                ("L", pick, False),         # 1 descend to pick
                ("L", pick_clear, True),    # 2 lift to clearance (now carrying)
                ("J", place_clear, True),   # 3 traverse high above slot (carrying)
                ("L", place_pose, True),    # 4 descend to place (carrying)
                ("L", place_clear, False),  # 5 retract to clearance (released)
            ]
            # solve IK for each leg
            qs = []
            reachable = True
            local_seed = seed.copy()
            for _mt, pose, _carry in legs:
                res = self._solve_best(pose, local_seed)
                qs.append(res.q)
                local_seed = res.q
                if not res.success:
                    reachable = False
            st.reachable = reachable

            # hard world for the arm: static obstacles + other pallets + this
            # pallet's slab. The growing stack (`placed`) is passed separately
            # as soft "place" boxes (only the carried box is checked against it).
            world = CollisionWorld(self.static + [pallet_slab], self.radii)

            box_start = len(sim)         # first sim sample belonging to this box
            # dense sim + collision check leg by leg
            leg_hit = None
            for li in range(len(legs)):
                q0 = sim[-1] if li == 0 else qs[li - 1]
                q1 = qs[li]
                seg = self.planner.joint_move(q0, q1, self.opts.sim_steps)
                carrying = legs[li][2]
                carried_box = Box(half=carry_half, T=local_carry, name=f"box{k}") \
                    if carrying else None
                # append samples + record base index
                base_i = len(sim)
                sim.extend(seg[1:].tolist())
                if leg_hit is None:
                    for j in range(1, len(seg)):
                        res = world.check(self.kin, seg[j], margin=self.opts.margin,
                                          carried=carried_box,
                                          carried_obstacles=self.static)
                        if res.hit:
                            leg_hit = (res, base_i + j - 1)
                            break
                # timeline events: attach carried box on leg 2 start, reveal + drop
                if li == 2:
                    events[base_i] = ("carry", local_carry.copy(), carry_half.copy())
                if li == 5:
                    events[base_i] = ("drop_reveal", k + 1)  # release + show box k

            if leg_hit is not None:
                st.collided = True
                res = leg_hit[0]
                st.detail = f"{res.link} ↔ {res.box} (d={res.distance*1000:.0f} mm)"

            # export steps (always emitted so a program can still be produced)
            self._emit_steps(steps, legs, qs)

            report.statuses.append(st)
            if (not st.reachable or st.collided) and report.first_failure < 0:
                report.first_failure = k
                report.ok = False
                report.first_fail_sample = (leg_hit[1] if (st.collided and leg_hit)
                                            else box_start)
                report.message = self._failure_message(k, place, st)

            placed.append(place.to_box(f"{self.spec.name}:box{k}"))
            seed = qs[-1]

        if report.ok:
            report.message = (f"All {len(placements)} boxes reachable and "
                              f"collision-free ({self.spec.layers} layers).")
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
        o = self.opts
        names = ["pick approach", "pick", "lift", "place approach", "place", "retract"]

        def move(mt, pose, q, name):
            if mt == "J":
                steps.append(ProgramStep(StepType.MOVEJ, name=name, q=list(map(float, q)),
                                         speed=o.speed_j, accel=o.accel_j))
            else:
                steps.append(ProgramStep(StepType.MOVEL, name=name,
                                         pose=list(map(float, pose)),
                                         speed=o.speed_l, accel=o.accel_l))

        # 0,1
        move(*legs[0][:2], qs[0], names[0])
        move(*legs[1][:2], qs[1], names[1])
        steps.append(ProgramStep(StepType.GRIPPER_CLOSE, name="grip"))
        # 2,3,4
        move(*legs[2][:2], qs[2], names[2])
        move(*legs[3][:2], qs[3], names[3])
        move(*legs[4][:2], qs[4], names[4])
        steps.append(ProgramStep(StepType.GRIPPER_OPEN, name="release"))
        # 5
        move(*legs[5][:2], qs[5], names[5])
