"""
robot/scene.py
==================================================================
The shared **world model** around the robot: user-created collision
obstacles plus palletizer pallets. One :class:`SceneModel` instance is
owned by the main window; the Scene/Palletizer panel edits it, and the 3D
viewport renders whatever it holds.

Everything is base-frame metres. The model is deliberately Qt-aware (it
emits ``changed``) in the same spirit as :mod:`robot.ur_bridge`, so views
can refresh reactively; the geometry itself lives in the Qt-free
:mod:`robot.collision` / :mod:`robot.palletizer` modules.
==================================================================
"""
from __future__ import annotations

import copy
from typing import List

import numpy as np
from PySide6.QtCore import QObject, Signal

from robot.collision import Box
from robot.palletizer import PalletSpec, generate_placements


class SceneModel(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.obstacles: List[Box] = []
        self.pallets: List[PalletSpec] = []

    # ---- obstacles --------------------------------------------------------
    def add_obstacle(self, box: Box) -> None:
        self.obstacles.append(box)
        self.changed.emit()

    def remove_obstacle(self, i: int) -> None:
        if 0 <= i < len(self.obstacles):
            self.obstacles.pop(i)
            self.changed.emit()

    def duplicate_obstacle(self, i: int, offset=(0.1, 0.0, 0.0)) -> None:
        if 0 <= i < len(self.obstacles):
            b = copy.deepcopy(self.obstacles[i])
            b.T = b.T.copy()
            b.T[:3, 3] = b.T[:3, 3] + np.asarray(offset, float)
            b.name = self._unique(b.name, [o.name for o in self.obstacles])
            self.obstacles.append(b)
            self.changed.emit()

    def add_pedestal(self, kin) -> None:
        """A rectangular base directly under the robot (mounts it higher)."""
        reach = 0.4
        try:
            reach = max(0.3, float(np.sum(np.abs(kin.a))) * 0.5)
        except Exception:                              # noqa: BLE001
            pass
        size = np.array([reach, reach, 0.3])
        # top face at z=0 (robot base sits on it) → centre below the base
        box = Box.from_size_center(size, [0.0, 0.0, -size[2] / 2.0],
                                   name=self._unique("pedestal",
                                                     [o.name for o in self.obstacles]),
                                   kind="pedestal")
        self.add_obstacle(box)

    # ---- pallets ----------------------------------------------------------
    def add_pallet(self, spec: PalletSpec) -> None:
        spec.name = self._unique(spec.name, [p.name for p in self.pallets])
        self.pallets.append(spec)
        self.changed.emit()

    def remove_pallet(self, i: int) -> None:
        if 0 <= i < len(self.pallets):
            self.pallets.pop(i)
            self.changed.emit()

    def duplicate_pallet(self, i: int, offset=(0.0, 0.7, 0.0)) -> None:
        """Copy/paste a whole pallet setup at an offset (multiple pallets)."""
        if 0 <= i < len(self.pallets):
            spec = copy.deepcopy(self.pallets[i])
            spec.T = spec.T.copy()
            spec.T[:3, 3] = spec.T[:3, 3] + np.asarray(offset, float)
            spec.name = self._unique(spec.name, [p.name for p in self.pallets])
            self.pallets.append(spec)
            self.changed.emit()

    def move_pallet(self, i: int, x=None, y=None, z=None, yaw_deg=None) -> None:
        """Absolutely set a pallet's position/yaw (whole stack moves with it)."""
        if not (0 <= i < len(self.pallets)):
            return
        spec = self.pallets[i]
        T = spec.T.copy()
        if x is not None:
            T[0, 3] = x
        if y is not None:
            T[1, 3] = y
        if z is not None:
            T[2, 3] = z
        if yaw_deg is not None:
            a = np.radians(yaw_deg)
            T[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0],
                                  [np.sin(a), np.cos(a), 0],
                                  [0, 0, 1]])
        spec.T = T
        self.changed.emit()

    # ---- direct manipulation (3D scene editing) --------------------------
    def move_obstacle(self, i: int, x=None, y=None, z=None) -> None:
        if not (0 <= i < len(self.obstacles)):
            return
        T = self.obstacles[i].T.copy()
        if x is not None:
            T[0, 3] = x
        if y is not None:
            T[1, 3] = y
        if z is not None:
            T[2, 3] = z
        self.obstacles[i].T = T
        self.changed.emit()

    def resize_obstacle(self, i: int, size_xyz) -> None:
        if 0 <= i < len(self.obstacles):
            self.obstacles[i].half = np.asarray(size_xyz, float) / 2.0
            self.changed.emit()

    def resize_pallet(self, i: int, size_xyz) -> None:
        if 0 <= i < len(self.pallets):
            self.pallets[i].size = np.asarray(size_xyz, float).reshape(3)
            self.changed.emit()

    # ---- collision assembly ----------------------------------------------
    def static_boxes(self, exclude_pallet: int = -1) -> List[Box]:
        """
        Obstacles + every *other* pallet's slab and fully-stacked boxes, i.e.
        the fixed world seen by the pallet at index ``exclude_pallet`` while it
        is being simulated.
        """
        boxes: List[Box] = [b for b in self.obstacles if b.enabled]
        for j, spec in enumerate(self.pallets):
            if j == exclude_pallet:
                continue
            boxes.append(spec.pallet_box())
            for k, pl in enumerate(generate_placements(spec)):
                boxes.append(pl.to_box(f"{spec.name}:box{k}"))
        return boxes

    def all_collision_boxes(self) -> List[Box]:
        """Everything solid in the scene (obstacles + all slabs + all boxes)."""
        return self.static_boxes(exclude_pallet=-1)

    # ---- render specs -----------------------------------------------------
    def render_specs(self) -> List[dict]:
        """Static render list for the viewport (obstacles + pallet slabs).

        Each spec carries a ``ref`` (('obstacle', i) / ('pallet', i)) so the
        viewport can map a picked actor back to the scene item the user clicked.
        """
        specs = []
        for i, b in enumerate(self.obstacles):
            specs.append(dict(T=b.T, half=b.half, name=b.name,
                              color="#bf616a" if b.kind != "pedestal" else "#4c566a",
                              opacity=0.35, ref=("obstacle", i)))
        for i, spec in enumerate(self.pallets):
            slab = spec.pallet_box()
            specs.append(dict(T=slab.T, half=slab.half, name=slab.name,
                              color=spec.color, opacity=0.65, ref=("pallet", i)))
        return specs

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _unique(name: str, taken) -> str:
        if name not in taken:
            return name
        base = name.rstrip("0123456789 ").strip() or "item"
        i = 2
        while f"{base} {i}" in taken:
            i += 1
        return f"{base} {i}"
