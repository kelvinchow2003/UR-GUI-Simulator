"""
robot/posture.py
==================================================================
A **collision-aware inverse-kinematics posture optimiser** — the piece that
makes the arm "smart" about *how* it reaches a pose.

A 6-axis UR can reach almost any TCP pose in several distinct joint
configurations ("branches"): elbow-up vs elbow-down, wrist flipped, the
base rotated to face the target from the front or the back. They all put
the tool in the same place, but some sweep the forearm straight through a
pallet stack while others tuck neatly around it. The stock damped-least-
squares solver just returns *the branch nearest its seed* — so whether the
arm collides is left to chance of the seed.

:class:`PostureOptimizer` instead **enumerates** the branches (by solving IK
from a diverse spread of seeds), keeps the reachable ones, and **scores**
each against the actual collision world:

    * collision clearance   — signed capsule-to-obstacle distance (the hard
                              filter: a colliding posture never beats a free
                              one), rewarding margin up to a cap;
    * singularity margin    — smallest singular value of the Jacobian, so the
                              arm keeps dexterity and avoids lock-up poses;
    * joint-limit margin    — distance of the nearest joint to its limit;
    * elbow-up bias         — the lowest link-origin height (a palletiser wants
                              the arm *above* the work, not dipping under it);
    * continuity            — how little the arm has to move from ``q_ref``.

The best-scoring configuration is returned, together with every candidate and
a short human-readable reason — so the same call can drive the planner *and*
explain, in the UI, why a given pick posture was chosen.

This is deterministic and runs in milliseconds: no training runs, no physics
engine. It reuses the app's own FK/IK (so the solutions are guaranteed
consistent with the digital twin) and the same :mod:`robot.collision` checker
the rest of the planner trusts.

Frames & units match the rest of the app: base-frame metres, TCP pose
``[x,y,z,rx,ry,rz]`` (axis-angle), joints in radians.
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from robot.kinematics import Kinematics
from robot.collision import Box, CollisionWorld


# ===========================================================================
#  Scoring weights
# ===========================================================================
@dataclass
class PostureWeights:
    """Relative importance of each posture quality term (see module docstring).

    Defaults are tuned for palletising: clearance and staying elbow-up dominate,
    with singularity/limit margins and motion continuity as tie-breakers."""
    clearance: float = 12.0        # per metre of collision clearance (capped)
    clearance_cap: float = 0.15    # clearance beyond this doesn't score higher
    singularity: float = 2.0       # per unit smallest-singular-value
    limit: float = 1.5             # per unit normalised joint-limit margin
    elbow_up: float = 3.0          # per metre of lowest link-origin height
    continuity: float = 0.5        # penalty per radian of motion from q_ref
    # A collision-free posture always outranks any colliding one, regardless of
    # the weighted score — this is the size of that guaranteed gap.
    collision_free_bonus: float = 1e6


# ===========================================================================
#  Results
# ===========================================================================
@dataclass
class PostureCandidate:
    q: np.ndarray
    reachable: bool
    in_limits: bool
    clearance: float = np.inf      # signed metres (>=0 collision-free); inf = no world
    sigma_min: float = 0.0         # smallest singular value of J (singularity margin)
    limit_margin: float = 0.0      # 0..1, nearest joint's normalised distance to a limit
    min_link_z: float = 0.0        # lowest moving-link origin height (elbow-up bias)
    motion: float = 0.0            # radians from q_ref
    score: float = -np.inf

    @property
    def collision_free(self) -> bool:
        return self.clearance >= 0.0


@dataclass
class PostureResult:
    ok: bool = False               # a reachable, in-limits posture was found
    best: Optional[PostureCandidate] = None
    candidates: List[PostureCandidate] = field(default_factory=list)
    reason: str = ""

    @property
    def q(self) -> Optional[np.ndarray]:
        return None if self.best is None else self.best.q


# ===========================================================================
#  Optimiser
# ===========================================================================
class PostureOptimizer:
    """Pick the smartest joint configuration for a TCP pose (see module docs)."""

    def __init__(self, kin: Kinematics, weights: Optional[PostureWeights] = None):
        self.kin = kin
        self.w = weights or PostureWeights()

    # ---- seed generation --------------------------------------------------
    def _seeds(self, pose: np.ndarray, q_ref: Optional[np.ndarray]) -> List[np.ndarray]:
        """A diverse spread of seed configurations, one per UR branch family.

        DLS converges to the solution *nearest* its seed, so seeding across the
        base-flip / shoulder / elbow-up-down / wrist-flip families is what makes
        the optimiser discover genuinely different postures rather than eight
        copies of one."""
        pose = np.asarray(pose, float)
        theta = float(np.arctan2(pose[1], pose[0]))     # base angle toward target
        seeds: List[np.ndarray] = []
        for base in (theta, self._wrap(theta + np.pi)):  # face target front / back
            for sh in (-0.7, -1.6):                     # shoulder spread
                for el in (1.4, -1.4):                    # elbow up / down
                    w1 = self._wrap(-(sh + el) - np.pi / 2)   # tool ~ downward
                    seeds.append(np.array([base, sh, el, w1, np.pi / 2, 0.0]))
        # Always include continuity + canonical seeds so a good nearby solution
        # and the standard "home" branch are never missed.
        if q_ref is not None:
            seeds.insert(0, np.asarray(q_ref, float).copy())
        seeds.append(self.kin._mid_config())
        return seeds

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    # ---- posture metrics --------------------------------------------------
    def _sigma_min(self, q: np.ndarray) -> float:
        """Smallest singular value of the Jacobian — a singularity margin that,
        unlike |det(J)|, degrades gracefully and is comparable across poses."""
        try:
            J = self.kin.jacobian(q)
            s = np.linalg.svd(J, compute_uv=False)
            return float(s[-1])
        except Exception:                               # noqa: BLE001
            return 0.0

    def _limit_margin(self, q: np.ndarray) -> float:
        """Nearest joint's distance to its limit, normalised by joint range
        (0 = at a limit, ~0.5 = dead-centre). Higher is safer."""
        lo, hi = self.kin.q_min, self.kin.q_max
        rng = np.maximum(hi - lo, 1e-6)
        margin = np.minimum(q - lo, hi - q) / rng
        return float(np.min(margin))

    def _min_link_z(self, q: np.ndarray) -> float:
        frames = self.kin.fk_frames(np.asarray(q, float))
        return float(min(f[2, 3] for f in frames[1:]))  # skip the fixed base

    # ---- the solve --------------------------------------------------------
    def solve(self, pose: np.ndarray,
              world: Optional[CollisionWorld] = None,
              q_ref: Optional[np.ndarray] = None,
              margin: float = 0.0,
              carried: Optional[Box] = None,
              carried_obstacles: Optional[List[Box]] = None,
              max_iter: int = 100,
              extra_seeds: Optional[List[np.ndarray]] = None) -> PostureResult:
        """Return the best joint configuration reaching ``pose``.

        ``world`` (optional) makes the choice collision-aware: candidates that
        clear the world always outrank ones that don't, and among the clear ones
        more margin scores higher. Without a world, selection falls back to
        singularity / limit / elbow-up / continuity quality only.

        ``carried`` / ``carried_obstacles`` mirror :meth:`CollisionWorld.check`
        for evaluating a posture while holding a gripped box.
        """
        # Fast path: if the continuity seed's own branch already reaches the pose
        # collision-free, keep it — no need to enumerate every branch. This makes
        # the common "nothing in the way" case about as cheap as a single IK
        # solve, and reserves the full multi-branch search for when the current
        # posture actually collides (exactly when re-posturing is worth it).
        if q_ref is not None and world is not None:
            r0 = self.kin.inverse(pose, q_init=np.asarray(q_ref, float),
                                  max_iter=max_iter)
            if r0.success and self.kin.in_limits(r0.q):
                c0 = world.check(self.kin, r0.q, margin=margin, carried=carried,
                                 carried_obstacles=carried_obstacles)
                if c0.distance >= 0.0:                  # clears the world already
                    cand = PostureCandidate(
                        q=r0.q, reachable=True, in_limits=True,
                        clearance=float(c0.distance),
                        sigma_min=self._sigma_min(r0.q),
                        limit_margin=self._limit_margin(r0.q),
                        min_link_z=self._min_link_z(r0.q), motion=0.0)
                    cand.score = self._score(cand, has_world=True)
                    return PostureResult(
                        ok=True, best=cand, candidates=[cand],
                        reason=f"kept current branch (collision-free, clearance "
                               f"{c0.distance * 1000:.0f} mm)")

        seeds = self._seeds(pose, q_ref)
        if extra_seeds:
            seeds = [np.asarray(s, float) for s in extra_seeds] + seeds

        candidates: List[PostureCandidate] = []
        for seed in seeds:
            res = self.kin.inverse(pose, q_init=seed, max_iter=max_iter)
            if not res.success:
                continue
            q = res.q
            if self._is_duplicate(q, candidates):
                continue
            cand = PostureCandidate(
                q=q,
                reachable=True,
                in_limits=self.kin.in_limits(q),
                sigma_min=self._sigma_min(q),
                limit_margin=self._limit_margin(q),
                min_link_z=self._min_link_z(q),
                motion=(0.0 if q_ref is None
                        else float(np.linalg.norm(self._wrap_vec(q - np.asarray(q_ref, float))))),
            )
            if world is not None:
                r = world.check(self.kin, q, margin=margin, carried=carried,
                                carried_obstacles=carried_obstacles)
                cand.clearance = float(r.distance)
            cand.score = self._score(cand, has_world=world is not None)
            candidates.append(cand)

        # Rank: valid postures first, then by score. A colliding posture can
        # still be returned (best-effort) when nothing collision-free was found.
        valid = [c for c in candidates if c.in_limits]
        pool = valid or candidates
        pool.sort(key=lambda c: c.score, reverse=True)
        result = PostureResult(ok=bool(valid), candidates=pool)
        if pool:
            result.best = pool[0]
            result.reason = self._explain(pool[0], has_world=world is not None,
                                           n=len(pool))
        else:
            result.reason = "No reachable configuration found for this pose."
        return result

    # ---- scoring & helpers ------------------------------------------------
    def _score(self, c: PostureCandidate, has_world: bool) -> float:
        w = self.w
        s = 0.0
        if has_world and np.isfinite(c.clearance):
            s += w.clearance * min(max(c.clearance, -w.clearance_cap), w.clearance_cap)
            if c.collision_free:
                s += w.collision_free_bonus
        s += w.singularity * c.sigma_min
        s += w.limit * c.limit_margin
        s += w.elbow_up * c.min_link_z
        s -= w.continuity * c.motion
        return s

    def _is_duplicate(self, q: np.ndarray, existing: List[PostureCandidate],
                      tol: float = 0.12) -> bool:
        for c in existing:
            if float(np.max(np.abs(self._wrap_vec(q - c.q)))) < tol:
                return True
        return False

    @staticmethod
    def _wrap_vec(dq: np.ndarray) -> np.ndarray:
        return (dq + np.pi) % (2 * np.pi) - np.pi

    def _explain(self, c: PostureCandidate, has_world: bool, n: int) -> str:
        bits = [f"chose 1 of {n} reachable posture(s)"]
        if has_world and np.isfinite(c.clearance):
            if c.collision_free:
                bits.append(f"collision-free (clearance {c.clearance * 1000:.0f} mm)")
            else:
                bits.append(f"⚠ best available still collides "
                            f"({-c.clearance * 1000:.0f} mm into an obstacle)")
        bits.append(f"singularity margin σ={c.sigma_min:.3f}")
        bits.append(f"joint-limit margin {c.limit_margin * 100:.0f}%")
        return "; ".join(bits)


# ---------------------------------------------------------------------------
#  Standalone demo: a box tucked against a wall, reachable only by the right
#  posture. Proves the optimiser finds a collision-free branch a naive seed
#  would miss. Run:  python -m robot.posture
# ---------------------------------------------------------------------------
def _demo() -> int:
    import sys
    from robot.ur_models import get_model
    from robot.kinematics import matrix_to_rotvec
    from robot.collision import robot_capsules, link_radii
    import numpy as _np
    try:
        sys.stdout.reconfigure(encoding="utf-8")        # allow σ / — glyphs
    except Exception:                                   # noqa: BLE001
        pass

    kin = Kinematics(get_model("UR10e"))
    opt = PostureOptimizer(kin)
    radii = link_radii(kin)

    # A pick pose in front of the robot, tool pointing straight down.
    R_down = _np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    pose = _np.concatenate([_np.array([0.55, 0.10, 0.20]), matrix_to_rotvec(R_down)])

    # What the stock solver would do: one nominal seed → one branch.
    naive = kin.inverse(pose, q_init=kin._mid_config())

    # Put an obstacle *exactly where the naive arm's forearm sits*, so the naive
    # posture is guaranteed to collide and only a different branch can avoid it.
    caps = robot_capsules(kin, naive.q, radii)
    p0, p1, _, _ = caps[2]                               # forearm segment
    mid = (_np.asarray(p0) + _np.asarray(p1)) / 2.0
    obstacle = Box.from_size_center([0.10, 0.10, 0.10], mid, name="obstacle")
    world = CollisionWorld([obstacle], radii=radii)

    naive_clear = world.check(kin, naive.q).distance if naive.success else float("nan")

    # Smart: collision-aware posture optimisation.
    res = opt.solve(pose, world=world, q_ref=kin._mid_config(), margin=0.0)

    print("Posture optimiser demo — obstacle on the naive arm's path (UR10e)\n")
    print(f"  naive single-seed IK:  reachable={naive.success}, "
          f"clearance={naive_clear * 1000:+.0f} mm  "
          f"-> {'COLLIDES' if naive.success and naive_clear < 0 else 'ok'}")
    if res.best is not None:
        c = res.best
        print(f"  smart optimiser:       clearance={c.clearance * 1000:+.0f} mm  "
              f"-> {'collision-free' if c.collision_free else 'still collides'}")
        print(f"  {res.reason}")
        print(f"  postures evaluated:    {len(res.candidates)}")
        improved = (naive.success and naive_clear < 0 and c.collision_free)
        print(f"\n  {'✓ smart posture avoids the obstacle the naive one hit.' if improved else '(scenario did not trip the naive solver this run)'}")
        return 0 if c.collision_free else 2
    print("  smart optimiser:       no reachable posture found")
    return 1


if __name__ == "__main__":
    raise SystemExit(_demo())
