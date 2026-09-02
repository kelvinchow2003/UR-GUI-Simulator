"""
robot/scene_planner.py
==================================================================
A **headless, Qt-free** planner for a whole palletizing scene — the same
sequential logic the Scene panel runs when you click "Simulate", factored
out so it can be driven without the GUI (e.g. by the MCP server that lets
Claude Desktop build a simulation for you).

Given a robot, a set of obstacles, and a list of pallets, it plans each
pallet in order — while the boxes already stacked on earlier pallets stand
as obstacles — and returns the chained program plus a feasibility report.

Frames & units match the rest of the app: base-frame metres.
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from robot.kinematics import Kinematics
from robot.collision import Box, link_radii
from robot.palletizer import (
    PalletSpec, JobOptions, PalletJob, generate_placements,
)
from robot.program import Program, ProgramStep


@dataclass
class ScenePlanResult:
    ok: bool
    steps: List[ProgramStep] = field(default_factory=list)
    message: str = ""
    total_boxes: int = 0
    placed_boxes: int = 0          # boxes planned before the first failure (if any)


def _static_for(obstacles: List[Box], pallets: List[PalletSpec],
                target: int) -> List[Box]:
    """The fixed world seen while filling pallet ``target``: user obstacles, every
    *other* pallet's slab, and the fully-stacked boxes of earlier pallets — mirrors
    ``SceneModel.static_boxes_for_sequence(target, set(range(target)))``."""
    boxes: List[Box] = [b for b in obstacles if getattr(b, "enabled", True)]
    for j, spec in enumerate(pallets):
        if j == target:
            continue
        boxes.append(spec.pallet_box())
        if j < target:                                  # already filled
            boxes += [pl.to_box(f"{spec.name}:box{i}")
                      for i, pl in enumerate(generate_placements(spec))]
    return boxes


def plan_scene(kin: Kinematics, obstacles: List[Box], pallets: List[PalletSpec],
               opts: Optional[JobOptions] = None,
               q_start: Optional[np.ndarray] = None,
               radii=None) -> ScenePlanResult:
    """Plan every pallet in list order as one continuous job (stack role).

    Returns the chained :class:`ProgramStep` list and a plain-language feasibility
    message. Stops adding steps at the first pallet that proves infeasible (the
    same behaviour as the Scene panel), but still reports what was planned."""
    opts = opts or JobOptions()
    radii = radii if radii is not None else link_radii(kin)
    q_prev = (np.asarray(q_start, float) if q_start is not None
              else np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]))

    steps: List[ProgramStep] = []
    parts: List[str] = []
    ok = True
    total = sum(p.total_boxes() for p in pallets)
    placed = 0
    for k, spec in enumerate(pallets):
        static = _static_for(obstacles, pallets, k)
        job = PalletJob(kin, static, spec, opts, q_start=q_prev, radii=radii)
        p_steps, sim, _events, report = job.plan()
        steps += p_steps
        sim = np.asarray(sim, float)
        if len(sim):
            q_prev = sim[-1]
        parts.append(f"{spec.name}: {report.message}")
        placed += sum(1 for s in report.statuses if s.reachable and not s.collided)
        if not report.ok:
            ok = False
            break
    return ScenePlanResult(ok=ok, steps=steps, message="   |   ".join(parts),
                           total_boxes=total, placed_boxes=placed)
