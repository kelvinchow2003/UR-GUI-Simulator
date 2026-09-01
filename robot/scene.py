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

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        """The whole editable world (obstacles + pallets) as plain data."""
        return {"obstacles": [b.to_dict() for b in self.obstacles],
                "pallets": [p.to_dict() for p in self.pallets]}

    def load_dict(self, d: dict) -> None:
        """Replace the scene contents from a saved dict and notify views once."""
        self.obstacles = [Box.from_dict(x) for x in d.get("obstacles", [])]
        self.pallets = [PalletSpec.from_dict(x) for x in d.get("pallets", [])]
        self.changed.emit()

    def clear(self) -> None:
        self.obstacles = []
        self.pallets = []
        self.changed.emit()

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

    def add_pedestal(self, kin, height: float = 0.3) -> None:
        """A rectangular base directly under the robot that **mounts it higher**.

        The pedestal sits *on the floor* (bottom face at z=0) and stacks on top
        of any pedestal already there. The robot base is then raised by the total
        pedestal height (see :meth:`pedestal_height`, applied by the main window)
        so it stands on the pedestal rather than being buried in the floor.
        """
        reach = 0.4
        try:
            reach = max(0.3, float(np.sum(np.abs(kin.a))) * 0.5)
        except Exception:                              # noqa: BLE001
            pass
        z0 = self.pedestal_height()                    # top of any existing stack
        size = np.array([reach, reach, float(height)])
        # bottom face on the floor (or on the pedestal below) → centre half-up
        box = Box.from_size_center(size, [0.0, 0.0, z0 + size[2] / 2.0],
                                   name=self._unique("pedestal",
                                                     [o.name for o in self.obstacles]),
                                   kind="pedestal")
        self.add_obstacle(box)

    def pedestal_height(self) -> float:
        """Total height of the pedestal stack under the robot base (0 if none).

        Pedestals sit on the floor, so the tallest top face is the height the
        robot base must be lifted to stand on them.
        """
        tops = [float(b.T[2, 3] + b.half[2]) for b in self.obstacles
                if b.kind == "pedestal" and b.enabled]
        return max(tops) if tops else 0.0

    # ---- conveyors --------------------------------------------------------
    def add_conveyor(self, length: float, width: float, height: float,
                     x: float = 0.4, y: float = -0.4, yaw_deg: float = 0.0) -> Box:
        """A floor-standing conveyor: a solid box (``length``×``width``×``height``
        metres) sitting on the floor with its top face at ``height``.

        It is stored as a normal obstacle (``kind="conveyor"``) so it is a real
        collision volume for the arm and can be selected/dragged/resized like any
        box. Its **top surface** doubles as the box-spawn / pick surface — call
        :meth:`conveyor_pick_point` to get where the robot lifts a box from it.
        """
        a = np.radians(yaw_deg)
        R = np.array([[np.cos(a), -np.sin(a), 0.0],
                      [np.sin(a), np.cos(a), 0.0],
                      [0.0, 0.0, 1.0]])
        size = np.array([float(length), float(width), float(height)])
        centre = np.array([float(x), float(y), float(height) / 2.0])   # on floor
        box = Box.from_size_center(size, centre,
                                   name=self._unique("conveyor",
                                                     [o.name for o in self.obstacles]),
                                   kind="conveyor", R=R)
        self.add_obstacle(box)
        return box

    @staticmethod
    def conveyor_pick_point(conv: Box, box_height: float = 0.0) -> np.ndarray:
        """World point where a box on the conveyor top is gripped: the top-face
        centre, raised by ``box_height`` so the tool meets the *top* of a box
        resting on the belt (grip = box top-face centre, matching the pallet)."""
        top_centre = conv.T[:3, 3] + conv.T[:3, :3] @ np.array(
            [0.0, 0.0, float(conv.half[2])])
        return top_centre + np.array([0.0, 0.0, float(box_height)])

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

    def static_boxes_for_sequence(self, target: int, filled) -> List[Box]:
        """The fixed world seen while filling pallet ``target`` in a sequential
        multi-pallet run: every user obstacle, every *other* pallet's slab, and
        the fully-stacked boxes of the pallets already filled (``filled`` = set
        of pallet indices completed before this one).

        Unlike :meth:`static_boxes`, the *future* pallets contribute only their
        (empty) slabs, not phantom boxes — so when the robot fills pallet 2 it
        must avoid the real boxes already on pallet 1, and errors if it can't."""
        filled = set(filled or ())
        boxes: List[Box] = [b for b in self.obstacles if b.enabled]
        for j, spec in enumerate(self.pallets):
            if j != target:
                boxes.append(spec.pallet_box())        # every other slab is solid
            if j in filled:
                for k, pl in enumerate(generate_placements(spec)):
                    boxes.append(pl.to_box(f"{spec.name}:box{k}"))
        return boxes

    # ---- render specs -----------------------------------------------------
    def render_specs(self) -> List[dict]:
        """Static render list for the viewport (obstacles + pallet slabs).

        Each spec carries a ``ref`` (('obstacle', i) / ('pallet', i)) so the
        viewport can map a picked actor back to the scene item the user clicked.
        """
        specs = []
        _obstacle_colors = {"pedestal": "#4c566a", "conveyor": "#434c5e"}
        for i, b in enumerate(self.obstacles):
            specs.append(dict(T=b.T, half=b.half, name=b.name, kind=b.kind,
                              color=_obstacle_colors.get(b.kind, "#bf616a"),
                              opacity=0.9 if b.kind == "conveyor" else 0.35,
                              ref=("obstacle", i)))
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
