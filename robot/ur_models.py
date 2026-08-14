"""
robot/ur_models.py
==================================================================
Static specification table for every supported Universal Robot.

A single :class:`URModel` dataclass captures everything the rest of the
application needs to treat any arm generically:

    * Denavit–Hartenberg parameters  (for the built-in FK/IK fallback)
    * per-joint position / velocity limits
    * default TCP payload & reach
    * mesh file stems for the 3D digital twin
    * dashboard/RTDE-relevant metadata (generation: CB3 vs e-Series)

DH values are the standard published UR kinematics (metres / radians).
They are *nominal*; a real robot carries small factory calibration
deltas exposed over RTDE, which the bridge can layer on top later.
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
import math

import numpy as np


class Generation(str, Enum):
    """UR controller generation — changes some dashboard semantics."""
    CB3 = "CB3"
    E_SERIES = "e-Series"


@dataclass(frozen=True)
class URModel:
    """Immutable kinematic + physical description of one UR arm."""

    name: str                     # e.g. "UR10e"
    family: str                   # e.g. "UR10"
    generation: Generation
    payload_kg: float
    reach_mm: float

    # Standard DH parameters (order: base -> wrist3). Units: metres, radians.
    a: List[float]                # link lengths
    d: List[float]                # link offsets
    alpha: List[float]            # link twists

    # Joint limits (radians). UR joints are nominally +/- 2*pi.
    q_min: List[float] = field(default_factory=lambda: [-2 * math.pi] * 6)
    q_max: List[float] = field(default_factory=lambda: [2 * math.pi] * 6)

    # Max joint speeds (rad/s) — used to clamp jogging & time trajectories.
    qd_max: List[float] = field(default_factory=lambda: [math.radians(180)] * 6)

    # Mesh file stems expected under assets/meshes/<family>/ (base..wrist3 + tool)
    mesh_stems: List[str] = field(
        default_factory=lambda: [
            "base", "shoulder", "upperarm", "forearm",
            "wrist1", "wrist2", "wrist3",
        ]
    )

    JOINT_NAMES = ("Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3")

    @property
    def dh(self) -> np.ndarray:
        """DH table as an (6, 3) array of [a, d, alpha] rows."""
        return np.column_stack((self.a, self.d, self.alpha))


# ---------------------------------------------------------------------------
# Registry
#
# DH parameters below are the widely published UR values. e-Series and CB3
# variants of the same reach share identical nominal link geometry, so we
# reuse one geometry block and only vary payload / generation metadata.
# ---------------------------------------------------------------------------
_PI = math.pi
_HALF_PI = math.pi / 2

# geometry: (a[], d[], alpha[]) per family ---------------------------------
_GEOMETRY: Dict[str, dict] = {
    "UR3": dict(
        a=[0, -0.24365, -0.21325, 0, 0, 0],
        d=[0.1519, 0, 0, 0.11235, 0.08535, 0.0819],
        alpha=[_HALF_PI, 0, 0, _HALF_PI, -_HALF_PI, 0],
        reach=500,
    ),
    "UR5": dict(
        a=[0, -0.425, -0.39225, 0, 0, 0],
        d=[0.089159, 0, 0, 0.10915, 0.09465, 0.0823],
        alpha=[_HALF_PI, 0, 0, _HALF_PI, -_HALF_PI, 0],
        reach=850,
    ),
    "UR10": dict(
        a=[0, -0.612, -0.5723, 0, 0, 0],
        d=[0.1273, 0, 0, 0.163941, 0.1157, 0.0922],
        alpha=[_HALF_PI, 0, 0, _HALF_PI, -_HALF_PI, 0],
        reach=1300,
    ),
    "UR16": dict(
        a=[0, -0.4784, -0.36, 0, 0, 0],
        d=[0.1807, 0, 0, 0.17415, 0.11985, 0.11655],
        alpha=[_HALF_PI, 0, 0, _HALF_PI, -_HALF_PI, 0],
        reach=900,
    ),
    "UR20": dict(
        a=[0, -0.8620, -0.7287, 0, 0, 0],
        d=[0.2363, 0, 0, 0.2010, 0.1593, 0.1543],
        alpha=[_HALF_PI, 0, 0, _HALF_PI, -_HALF_PI, 0],
        reach=1750,
    ),
    "UR30": dict(
        a=[0, -0.6370, -0.5037, 0, 0, 0],
        d=[0.2363, 0, 0, 0.2010, 0.1593, 0.1543],
        alpha=[_HALF_PI, 0, 0, _HALF_PI, -_HALF_PI, 0],
        reach=1300,
    ),
}

# payload per family (kg) — same across CB3/e-Series where both exist -------
_PAYLOAD = {"UR3": 3, "UR5": 5, "UR10": 10, "UR16": 16, "UR20": 20, "UR30": 30}

# joint speed limits (deg/s) — small arms are faster ------------------------
_QD_MAX_DEG = {
    "UR3": 180, "UR5": 180, "UR10": 120, "UR16": 120, "UR20": 120, "UR30": 120,
}


def _build_registry() -> Dict[str, URModel]:
    models: Dict[str, URModel] = {}
    for family, geo in _GEOMETRY.items():
        # UR16/20/30 are e-Series only; UR3/5/10 exist as both CB3 and e-Series.
        gens = [Generation.E_SERIES]
        if family in ("UR3", "UR5", "UR10"):
            gens = [Generation.CB3, Generation.E_SERIES]

        for gen in gens:
            suffix = "e" if gen is Generation.E_SERIES else ""
            name = f"{family}{suffix}"
            qd = math.radians(_QD_MAX_DEG[family])
            models[name] = URModel(
                name=name,
                family=family,
                generation=gen,
                payload_kg=_PAYLOAD[family],
                reach_mm=geo["reach"],
                a=list(geo["a"]),
                d=list(geo["d"]),
                alpha=list(geo["alpha"]),
                qd_max=[qd] * 6,
            )
    return models


UR_MODELS: Dict[str, URModel] = _build_registry()

#: Convenient ordered list for populating a combo box.
MODEL_NAMES: List[str] = list(UR_MODELS.keys())

DEFAULT_MODEL = "UR5e"


def get_model(name: str) -> URModel:
    """Look up a model by name, raising a clear error for unknown arms."""
    try:
        return UR_MODELS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown UR model '{name}'. Known: {', '.join(MODEL_NAMES)}"
        ) from exc
