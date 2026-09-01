"""
robot/robodk_export.py
==================================================================
Export a palletizing scene + planned program into **RoboDK**, so the whole
pick→place job can be re-simulated on RoboDK's own kinematic / collision
engine to discover the *limits* of a cell:

    * reachability     — targets RoboDK's IK cannot solve (out of the reach
                         envelope, or blocked by joint limits);
    * collisions       — arm vs pallet slab, vs the growing box stack, vs
                         user obstacles, checked by RoboDK's independent
                         mesh-collision engine;
    * cycle time       — RoboDK's own time estimate for the full sequence;
    * singularities    — flagged by RoboDK along the Cartesian legs.

Nothing in the rest of the app depends on RoboDK. This module imports it
lazily and degrades with a clear message when it (or the RoboDK desktop
app) is absent — matching the "every heavy dependency is optional" spirit
of the project.

Frames & units
--------------
The app works in **base-frame metres**, TCP pose ``[x,y,z,rx,ry,rz]``
(axis-angle). RoboDK works in **station-frame millimetres**, degrees for
joints, and 4x4 homogeneous ``Mat`` poses. All conversions live here:

    * the robot base is placed at ``kin.base_pose()`` (pedestal height and
      any repositioning included), translation scaled m→mm;
    * ``MoveJ`` legs export as **joint targets** (exact joints, rad→deg) so
      the posture the planner chose is reproduced verbatim;
    * ``MoveL`` legs export as **Cartesian targets** in the station frame,
      so RoboDK solves its *own* IK and can independently flag an
      unreachable or singular pose — which is the whole point of the check.

Usage
-----
Programmatic (e.g. from the Scene panel's "Export to RoboDK" button)::

    from robot.robodk_export import export_scene_to_robodk
    report = export_scene_to_robodk(kin, obstacles, pallets, steps, placed)
    print(report.text)

Standalone connection self-test (RoboDK must be installed & running)::

    python -m robot.robodk_export
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from robot.kinematics import Kinematics, pose_to_matrix
from robot.program import ProgramStep, StepType


class RoboDKUnavailable(RuntimeError):
    """Raised when the ``robodk`` package or the RoboDK app can't be reached."""


# ---------------------------------------------------------------------------
#  Lazy import — keeps the whole app importable without robodk installed.
# ---------------------------------------------------------------------------
def _import_robodk():
    try:
        from robodk import robolink, robomath          # type: ignore
    except Exception as exc:                            # noqa: BLE001
        raise RoboDKUnavailable(
            "The 'robodk' Python package is not installed. Install it with:\n"
            "    pip install robodk\n"
            "and install the RoboDK desktop app from https://robodk.com/download"
        ) from exc
    return robolink, robomath


# ---------------------------------------------------------------------------
#  Unit / frame conversion
# ---------------------------------------------------------------------------
_M2MM = 1000.0

# Every station item this exporter creates is named with this prefix, so a
# re-export can delete *only* what it made before (never the robot, the user's
# own frames, or anything else) and rebuild from a clean slate.
_TAG = "URGUISim"


def _strip_html(s: str) -> str:
    """RoboDK status messages come with HTML markup; render them as plain text."""
    import re
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(s.split())


def _mat_from_T(robomath, T: np.ndarray):
    """A 4x4 base-frame-metres transform -> a RoboDK ``Mat`` in millimetres."""
    T = np.asarray(T, float).reshape(4, 4)
    rows = []
    for r in range(4):
        row = [float(T[r, 0]), float(T[r, 1]), float(T[r, 2])]
        # scale only the translation column (m -> mm); rotation is unitless
        row.append(float(T[r, 3]) * (_M2MM if r < 3 else 1.0))
        rows.append(row)
    return robomath.Mat(rows)


def _mat_from_pose(robomath, pose: Sequence[float]):
    """A UR TCP pose ``[x,y,z,rx,ry,rz]`` (metres/rotvec) -> RoboDK ``Mat`` (mm).

    Reuses the app's tested ``pose_to_matrix`` for the axis-angle → matrix step
    so the rotation convention is guaranteed identical to what the planner and
    digital twin used."""
    return _mat_from_T(robomath, pose_to_matrix(np.asarray(pose, float)))


# ---------------------------------------------------------------------------
#  Box geometry — RoboDK has no "add primitive" call, so build the 12 triangles
#  of an axis-aligned cuboid centred at the origin (local frame), in mm.
# ---------------------------------------------------------------------------
def _box_vertices(size_mm: Sequence[float]) -> List[tuple]:
    """The 36 triangle vertices (12 triangles) of a centred cuboid, in order."""
    hx, hy, hz = (float(s) / 2.0 for s in size_mm)
    v = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),  # bottom
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),      # top
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),        # bottom (-z)
        (4, 6, 5), (4, 7, 6),        # top    (+z)
        (0, 5, 1), (0, 4, 5),        # -y
        (3, 2, 6), (3, 6, 7),        # +y
        (1, 6, 2), (1, 5, 6),        # +x
        (0, 3, 7), (0, 7, 4),        # -x
    ]
    verts: List[tuple] = []
    for a, b, c in faces:
        verts.extend((v[a], v[b], v[c]))
    return verts


def _box_shape_mat(robomath, size_mm: Sequence[float]):
    """A RoboDK ``Mat`` of shape 3xN (columns = triangle vertices) for AddShape."""
    verts = _box_vertices(size_mm)                      # N tuples of (x,y,z)
    rows = [[vt[axis] for vt in verts] for axis in range(3)]   # 3 rows, N cols
    return robomath.Mat(rows)


# ---------------------------------------------------------------------------
#  Result report
# ---------------------------------------------------------------------------
@dataclass
class ExportReport:
    ok: bool = False
    text: str = ""
    robot_name: str = ""
    n_targets: int = 0
    n_valid: int = 0                    # instructions RoboDK could execute
    valid_ratio: float = 0.0           # 0..1 from Program.Update
    cycle_time_s: float = 0.0
    n_collisions: int = -1             # -1 = not evaluated
    messages: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Robot-name resolution: app model names vs RoboDK library filenames.
# ---------------------------------------------------------------------------
def _family(name: str) -> str:
    """Reduce a robot name to its UR family for matching: 'UR10e', 'UR10',
    'Universal Robots UR10e' → 'UR10'. Same family ⇒ identical reach & geometry."""
    import re
    m = re.search(r"UR\s*0*([0-9]+)", name, flags=re.IGNORECASE)
    return f"UR{m.group(1)}" if m else name.strip().upper()


def _robot_name_candidates(model_name: str) -> List[str]:
    """Plausible RoboDK robot names for an app model (e.g. 'UR20e').

    RoboDK's library mostly matches the app's names ('UR5e', 'UR10e', 'UR16e'),
    but the largest arms are listed without the 'e' suffix ('UR20', 'UR30')."""
    name = model_name.strip()
    cands = [name]
    if name.endswith("e"):
        cands.append(name[:-1])                 # UR20e -> UR20
    else:
        cands.append(name + "e")
    # de-dup preserving order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ===========================================================================
#  Exporter
# ===========================================================================
class RoboDKExporter:
    """Build a RoboDK station from a planned palletizing scene and analyse it."""

    def __init__(self, kin: Kinematics, robot_name: Optional[str] = None):
        self.kin = kin
        self.robot_name = robot_name or getattr(kin.model, "name", "UR5e")
        self.robolink, self.robomath = _import_robodk()
        self.RDK = None
        self.robot = None
        self.tool = None
        self.frame = None
        self.warnings: List[str] = []

    # ---- connection & robot ----------------------------------------------
    def connect(self):
        try:
            self.RDK = self.robolink.Robolink()
            # a trivial call to confirm the link is actually alive
            self.RDK.Version()
        except Exception as exc:                        # noqa: BLE001
            raise RoboDKUnavailable(
                "Could not connect to the RoboDK application. Start RoboDK and "
                "leave it running, then export again."
            ) from exc
        return self.RDK

    def get_or_load_robot(self):
        """Find a UR already in the open station, else auto-load one from the local
        RoboDK library. Matching is by *family* (UR10e ↔ UR10 share identical reach
        and link geometry — only payload/generation differ), so a same-family robot
        gives accurate reach/joint-limit results and raises no warning. Falls back
        with a clear, actionable error."""
        RDK = self.RDK
        rl = self.robolink
        want = _family(self.robot_name)
        robots = RDK.ItemList(rl.ITEM_TYPE_ROBOT)
        # 1) a same-family robot already in the station? (best case)
        for r in robots:
            if _family(r.Name()) == want:
                self.robot = r
                return r
        # 2) auto-load a same-family robot from the local library folder.
        item = self._load_robot_from_library(want)
        if item is not None:
            self.robot = item
            return item
        # 3) any robot already in the station — motion still reproduces via joint
        #    targets, but reach/limit results are for THIS robot, so warn loudly.
        if robots:
            self.robot = robots[0]
            self.warnings.append(
                f"⚠ RoboDK station has '{self.robot.Name()}' but your app model is "
                f"'{self.robot_name}'. Reach/joint-limit results are for "
                f"'{self.robot.Name()}', not your robot. Add a '{self.robot_name}' "
                f"in RoboDK (Model Library) for accurate limits.")
            return self.robot
        raise RoboDKUnavailable(
            f"No robot found in the RoboDK station and no '{want}' family robot is "
            f"in your local RoboDK library. In RoboDK, open the Model Library "
            f"(online) and add a '{self.robot_name}', then export again."
        )

    def _load_robot_from_library(self, family: str):
        """AddFile the first ``*.robot`` in the local library whose filename is the
        same UR family. Returns the loaded robot item, or None if none found."""
        import glob
        import os
        RDK = self.RDK
        try:
            lib = RDK.getParam("PATH_LIBRARY") or ""
        except Exception:                               # noqa: BLE001
            lib = ""
        if not lib or not os.path.isdir(lib):
            return None
        for path in sorted(glob.glob(os.path.join(lib, "**", "*.robot"),
                                     recursive=True)):
            stem = os.path.splitext(os.path.basename(path))[0]
            if _family(stem) == family:
                item = RDK.AddFile(path)
                if item is not None and item.Valid():
                    return item
        return None

    def _set_robot_base(self, T_world_base: np.ndarray) -> bool:
        """Physically place the robot's base at ``T_world_base`` (base-frame metres,
        pedestal height included).

        RoboDK versions differ on how a robot base is positioned: some expose
        ``robot.setPoseBase`` on the robot, but many only let you move the robot's
        **base reference frame** (its parent item). We set that frame's *absolute*
        (station) pose, which lifts the robot regardless of the tree hierarchy, and
        fall back to any robot-level setter if the frame isn't where we expect."""
        rm, rl = self.robomath, self.robolink
        H = _mat_from_T(rm, T_world_base)
        base_frame = self.robot.Parent()
        if base_frame is not None and base_frame.Type() == rl.ITEM_TYPE_FRAME:
            for setter in ("setPoseAbs", "setPose"):
                if hasattr(base_frame, setter):
                    try:
                        getattr(base_frame, setter)(H)
                        return True
                    except Exception:                   # noqa: BLE001
                        continue
        for setter in ("setPoseBase", "setPoseAbs", "setPose"):
            if hasattr(self.robot, setter):
                try:
                    getattr(self.robot, setter)(H)
                    return True
                except Exception:                       # noqa: BLE001
                    continue
        return False

    def clear_previous(self) -> int:
        """Delete every item a *previous* export created (name starts with the
        tag), so re-exporting doesn't pile up duplicate frames/boxes/programs or
        leak stale collision state. Never touches the robot or user content."""
        RDK, rl = self.RDK, self.robolink
        removed = 0
        # Delete targets/programs before their parent frames, tools before robot.
        for typ in (rl.ITEM_TYPE_TARGET, rl.ITEM_TYPE_PROGRAM, rl.ITEM_TYPE_OBJECT,
                    rl.ITEM_TYPE_TOOL, rl.ITEM_TYPE_FRAME):
            try:
                items = RDK.ItemList(typ)
            except Exception:                           # noqa: BLE001
                continue
            for it in items:
                try:
                    if it.Valid() and it.Name().startswith(_TAG):
                        it.Delete()
                        removed += 1
                except Exception:                       # noqa: BLE001
                    continue
        return removed

    def setup_frames(self, prog_tcp: Optional[Sequence[float]] = None):
        """Place the robot base at ``kin.base_pose()``; add a world reference for
        Cartesian targets; add a tool matching the app's TCP offset."""
        RDK, rl, rm = self.RDK, self.robolink, self.robomath
        self.clear_previous()
        # Robot base transform = the app's world->base (pedestal height and any
        # repositioning included). Done first so target IK sees the lifted base.
        if not self._set_robot_base(self.kin.base_pose()):
            self.warnings.append("⚠ Could not set the robot base pose in RoboDK — "
                                 "any pedestal height may not be reflected.")
        # A single world reference frame at the station origin. All Cartesian
        # targets are expressed relative to it, matching the app's world frame.
        self.frame = RDK.AddFrame(f"{_TAG} world")
        self.frame.setPoseAbs(rm.transl(0, 0, 0))       # identity at station origin
        # tool: app TCP offset (tool0 -> TCP). Prefer the program's TCP if given.
        tcp_T = self.kin._tcp
        if prog_tcp is not None and np.any(np.asarray(prog_tcp, float)):
            tcp_T = pose_to_matrix(np.asarray(prog_tcp, float))
        self.tool = self.robot.AddTool(_mat_from_T(rm, tcp_T),
                                       tool_name=f"{_TAG} TCP")
        self.robot.setPoseTool(self.tool)
        self.robot.setPoseFrame(self.frame)

    # ---- scene geometry ---------------------------------------------------
    def add_scene(self, obstacles=None, pallets=None, placed=None):
        """Add pallet slabs, placed boxes, and user obstacles as RoboDK objects
        so RoboDK's collision engine has something real to test the arm against.

        * ``obstacles`` — list of :class:`robot.collision.Box`.
        * ``pallets``   — list of :class:`robot.palletizer.PalletSpec`.
        * ``placed``    — list of ``dict(T=4x4, half=(3,), color=?)`` (the exact
                          box placements the planner produced; from the scene
                          panel's plan result ``res['placed']``).
        """
        for b in (obstacles or []):
            if getattr(b, "enabled", True):
                obj = self._add_box(2.0 * b.half, b.T,
                                    f"{_TAG} {b.name or 'obstacle'}")
                # The robot base sits ON its pedestal, so a robot-vs-pedestal
                # overlap is by design, not a crash — disable that collision pair
                # so it doesn't show up as a permanent false positive.
                if getattr(b, "kind", "") == "pedestal":
                    self._disable_collision_with_robot(obj)
        for p in (pallets or []):
            slab = p.pallet_box()
            self._add_box(2.0 * slab.half, slab.T, f"{_TAG} {slab.name}")
        for i, spec in enumerate(placed or []):
            half = np.asarray(spec["half"], float)
            self._add_box(2.0 * half, spec["T"], f"{_TAG} box{i}")

    def _isolate_robot_collisions(self) -> None:
        """Disable collisions between our robot and any *other* robot in the
        station. Two arms parked at the origin (e.g. a leftover robot plus one we
        auto-loaded) otherwise overlap and report spurious collisions that have
        nothing to do with the palletizing motion we're checking."""
        RDK, rl = self.RDK, self.robolink
        try:
            others = [r for r in RDK.ItemList(rl.ITEM_TYPE_ROBOT)
                      if r.item != self.robot.item]
        except Exception:                               # noqa: BLE001
            others = []
        for other in others:
            try:
                RDK.setCollisionActivePair(rl.COLLISION_OFF, self.robot, other)
            except Exception:                           # noqa: BLE001
                pass

    def _disable_collision_with_robot(self, obj) -> None:
        """Turn off collision checking between the robot and one object (used for
        the pedestal the robot legitimately stands on)."""
        rl = self.robolink
        try:
            self.RDK.setCollisionActivePair(rl.COLLISION_OFF, self.robot, obj)
        except Exception:                               # noqa: BLE001
            pass                                        # older API / not critical

    def _add_box(self, size_m, T, name: str):
        """Add one oriented cuboid to the station at world transform ``T``."""
        RDK, rm = self.RDK, self.robomath
        size_mm = [float(s) * _M2MM for s in np.asarray(size_m, float)]
        obj = RDK.AddShape(_box_shape_mat(rm, size_mm))
        obj.setName(name)
        obj.setPose(_mat_from_T(rm, T))
        return obj

    # ---- program ----------------------------------------------------------
    def add_program(self, steps: List[ProgramStep], name: str = "Palletize"):
        """Turn the planned steps into a RoboDK program of targets + moves."""
        RDK, rl = self.RDK, self.robolink
        if not name.startswith(_TAG):
            name = f"{_TAG} {name}"
        prog = RDK.AddProgram(name, self.robot)
        prog.setPoseFrame(self.frame)
        prog.setPoseTool(self.tool)
        n_targets = 0
        for i, s in enumerate(steps):
            if not s.enabled:
                continue
            t = s.type
            if t is StepType.MOVEJ and s.q is not None:
                tgt = RDK.AddTarget(f"{_TAG} J{i}", self.frame, self.robot)
                tgt.setAsJointTarget()
                tgt.setJoints([np.degrees(v) for v in s.q])
                prog.MoveJ(tgt)
                n_targets += 1
            elif t in (StepType.MOVEL, StepType.MOVEP, StepType.PROCESS) and s.pose is not None:
                tgt = RDK.AddTarget(f"{_TAG} L{i}", self.frame, self.robot)
                tgt.setAsCartesianTarget()
                tgt.setPose(_mat_from_pose(self.robomath, s.pose))
                prog.MoveL(tgt)
                n_targets += 1
            elif t in (StepType.GRIPPER_OPEN, StepType.GRIPPER_CLOSE):
                prog.setDO(0, 0 if t is StepType.GRIPPER_OPEN else 1)
            elif t is StepType.SET_DO:
                prog.setDO(int(s.pin), 1 if s.value else 0)
            elif t is StepType.WAIT_DI:
                prog.waitDI(int(s.pin), 1)
            elif t is StepType.DELAY:
                prog.Pause(float(s.duration) * 1000.0)
            elif t is StepType.COMMENT:
                prog.RunInstruction(s.text or "", rl.INSTRUCTION_COMMENT)
        self.program = prog
        self.n_targets = n_targets
        return prog

    # ---- analysis ---------------------------------------------------------
    def analyse(self, check_collisions: bool = True) -> ExportReport:
        """Run RoboDK's own validity + timing pass and read back the limits."""
        RDK, rl = self.RDK, self.robolink
        rep = ExportReport(robot_name=self.robot.Name(), n_targets=self.n_targets)
        rep.messages.extend(self.warnings)
        if check_collisions:
            try:
                RDK.setCollisionActive(rl.COLLISION_ON)
            except Exception:                           # noqa: BLE001
                pass
            self._isolate_robot_collisions()
        # Program.Update -> [valid_instructions, program_time, program_distance,
        #                    valid_ratio(0..1), readable_message]
        try:
            coll_flag = rl.COLLISION_ON if check_collisions else rl.COLLISION_OFF
            upd = self.program.Update(coll_flag)
        except TypeError:
            upd = self.program.Update()
        try:
            rep.cycle_time_s = float(upd[1])
            rep.valid_ratio = float(upd[3])
            # Report the reachable move count consistently with the ratio RoboDK
            # returns (upd[0] counts *all* instructions, incl. I/O, so it doesn't
            # match the motion-target total — derive from the ratio instead).
            rep.n_valid = int(round(rep.valid_ratio * rep.n_targets))
            if len(upd) > 4 and upd[4]:
                rep.messages.append(_strip_html(str(upd[4])))
        except Exception:                               # noqa: BLE001
            pass
        if check_collisions:
            try:
                rep.n_collisions = int(RDK.Collisions())
            except Exception:                           # noqa: BLE001
                rep.n_collisions = -1
        rep.ok = (rep.valid_ratio >= 0.999
                  and (rep.n_collisions in (-1, 0)))
        rep.text = self._format(rep)
        return rep

    @staticmethod
    def _format(rep: ExportReport) -> str:
        lines = [f"RoboDK export — robot '{rep.robot_name}', {rep.n_targets} targets."]
        if rep.valid_ratio >= 0.999:
            lines.append(f"✓ All {rep.n_targets} moves valid on RoboDK's engine.")
        else:
            pct = rep.valid_ratio * 100.0
            failed = max(rep.n_targets - rep.n_valid, 1)
            lines.append(f"✗ {failed} of {rep.n_targets} moves failed on RoboDK "
                         f"({pct:.0f}% of the path valid) — a target is out of reach / "
                         f"at a joint limit, or a linear (MoveL) leg can't be done "
                         f"in a straight line (crosses a singularity).")
        if rep.n_collisions == 0:
            lines.append("✓ No collisions detected by RoboDK.")
        elif rep.n_collisions > 0:
            lines.append(f"✗ {rep.n_collisions} collision pair(s) detected by RoboDK.")
        if rep.cycle_time_s > 0:
            lines.append(f"Estimated cycle time: ~{rep.cycle_time_s:.1f}s (RoboDK).")
        lines += [f"  · {m}" for m in rep.messages if m]
        return "\n".join(lines)


# ===========================================================================
#  One-call convenience
# ===========================================================================
def export_scene_to_robodk(
    kin: Kinematics,
    obstacles=None,
    pallets=None,
    steps: Optional[List[ProgramStep]] = None,
    placed=None,
    robot_name: Optional[str] = None,
    program_name: str = "Palletize",
    prog_tcp: Optional[Sequence[float]] = None,
    check_collisions: bool = True,
) -> ExportReport:
    """Build the full station in RoboDK and return an :class:`ExportReport`.

    Raises :class:`RoboDKUnavailable` if RoboDK (package or app) is missing —
    the caller should catch it and surface the message to the user.
    """
    ex = RoboDKExporter(kin, robot_name=robot_name)
    ex.connect()
    ex.get_or_load_robot()
    ex.setup_frames(prog_tcp=prog_tcp)
    ex.add_scene(obstacles=obstacles, pallets=pallets, placed=placed)
    if steps:
        ex.add_program(steps, name=program_name)
        return ex.analyse(check_collisions=check_collisions)
    rep = ExportReport(ok=True, robot_name=ex.robot.Name())
    rep.text = f"RoboDK station built for '{rep.robot_name}' (no program steps)."
    return rep


# ---------------------------------------------------------------------------
#  Standalone connection self-test.
# ---------------------------------------------------------------------------
def _selftest() -> int:
    import sys
    from robot.ur_models import get_model
    try:                                               # Windows consoles default
        sys.stdout.reconfigure(encoding="utf-8")       # to cp1252 — allow ✓/✗/·
    except Exception:                                  # noqa: BLE001
        pass
    try:
        kin = Kinematics(get_model("UR10e"))
        # A minimal *joint-space* toy motion: two safe upright postures near the UR
        # "home" pose. Pure MoveJ has no IK/singularity ambiguity, so it stays valid
        # and collision-free on ANY UR that happens to be loaded — the point here is
        # to prove the export pipeline reaches RoboDK, not to validate a motion. The
        # Cartesian/IK path is exercised by a real palletizing export from the GUI.
        HALF_PI = float(np.pi / 2)
        q0 = [0.0, -HALF_PI, 0.0, -HALF_PI, 0.0, 0.0]
        q1 = [0.35, -HALF_PI, 0.0, -HALF_PI, 0.0, 0.0]

        ex = RoboDKExporter(kin)
        ex.connect()
        ex.get_or_load_robot()
        ex.setup_frames()
        steps = [
            ProgramStep(StepType.MOVEJ, q=q0),
            ProgramStep(StepType.GRIPPER_CLOSE),
            ProgramStep(StepType.MOVEJ, q=q1),
            ProgramStep(StepType.GRIPPER_OPEN),
        ]
        ex.add_program(steps, name="SelfTest")
        rep = ex.analyse()
        print("Pipeline reached RoboDK and built a station + program. "
              "RoboDK reports:\n")
        print(rep.text)
        print("\n(If RoboDK just opened on its own, that's expected — the robodk "
              "package auto-launches the app to build the station in it. This "
              "smoke test uses a generic toy motion; run a real palletizing export "
              "from the app for meaningful reach/collision limits.)")
        return 0                                        # pipeline completed
    except RoboDKUnavailable as exc:
        print("RoboDK not available:\n" + str(exc))
        return 1


if __name__ == "__main__":
    import os
    rc = _selftest()
    # The robodk socket keeps a background thread alive that can stall a normal
    # interpreter exit; flush and hard-exit so the CLI self-test returns promptly.
    import sys
    sys.stdout.flush()
    os._exit(rc)
