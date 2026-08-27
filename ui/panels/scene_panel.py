"""
ui/panels/scene_panel.py
==================================================================
Scene & Palletizer dock.

Two cooperating tools built on the shared :class:`robot.scene.SceneModel`:

    * **Obstacles** — create rectangular collision volumes (e.g. a pedestal
      under the robot, guarding, fixtures). The arm must not hit them; the
      collision checker (``robot.collision``) flags any breach in Run
      Offline Simulation and in the palletizer.

    * **Palletizer** — define a pallet (size + pose), a box size, a grip
      ("touch") point on the box, a base pattern and layer count. Preview
      the stack, then **Simulate palletization**: the twin runs the full
      pick→place cycle box-by-box while the checker verifies reachability
      and collisions, stopping at the first failure. Pallets can be copied,
      moved and deleted, and the whole job exported into the Program as a
      real, runnable UR program.

Everything is base-frame metres, shown to the user in millimetres.

Safety: this is a *feasibility aid*, not a safety certificate. The twin
uses nominal geometry (no cabling, real gripper, payload or controller
safety planes). Always dry-run on the real robot at reduced speed with the
E-stop in reach.
==================================================================
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QSpinBox, QListWidget, QComboBox,
    QMenu, QInputDialog, QCheckBox,
)

from robot.collision import Box, CollisionWorld, link_radii
from robot.palletizer import (
    PalletSpec, JobOptions, PalletJob, TransferJob, generate_placements,
)

_ROLES = ["stack", "source", "destination"]     # role_box index → PalletSpec.role


def _spin(lo, hi, val, dec=0, step=10.0, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class ScenePanel(QWidget):
    def __init__(self, scene, viewport, program_panel, kin, main_window,
                 parent=None):
        super().__init__(parent)
        self.scene = scene
        self.viewport = viewport
        self.program_panel = program_panel
        self.kin = kin
        self.main_window = main_window
        self._radii = None
        self._preview_specs = None          # currently previewed placements
        self._sel_ref = None                # ('obstacle'|'pallet', index) or None
        self._scene_ctl = None
        self._build()
        self.scene.changed.connect(self._render_scene)
        self._render_scene()
        self._install_scene_controller()

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        tip = QLabel("Tip: click an object in the 3D view to select it — drag to "
                     "move (Shift = up/down), Del to delete, right-click for "
                     "duplicate / resize.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#5e81ac;font-size:10px")
        root.addWidget(tip)

        # ---------- origin-axis gizmos ----------
        axrow = QHBoxLayout()
        axrow.addWidget(QLabel("Origin axes:"))
        self.ax_robot = QCheckBox("Robot")
        self.ax_pallet = QCheckBox("Pallet")
        self.ax_conveyor = QCheckBox("Conveyor")
        for c in (self.ax_robot, self.ax_pallet, self.ax_conveyor):
            c.setToolTip("Show a draggable 3-axis gizmo at each origin. Drag an "
                         "arrow to move along that axis; drag a ring to rotate "
                         "about it. Red=X, Green=Y, Blue=Z.")
            c.toggled.connect(self._rebuild_axes)
            axrow.addWidget(c)
        axrow.addStretch(1)
        root.addLayout(axrow)

        # ---------- obstacles ----------
        ob = QGroupBox("Obstacles  (collision volumes)")
        og = QGridLayout(ob)
        self.obs_list = QListWidget()
        self.obs_list.setMaximumHeight(90)
        og.addWidget(self.obs_list, 0, 0, 1, 4)
        og.addWidget(QLabel("Size L×W×H (mm)"), 1, 0)
        self.ob_l = _spin(1, 5000, 300); self.ob_w = _spin(1, 5000, 300)
        self.ob_h = _spin(1, 5000, 200)
        og.addWidget(self.ob_l, 1, 1); og.addWidget(self.ob_w, 1, 2)
        og.addWidget(self.ob_h, 1, 3)
        og.addWidget(QLabel("Centre X/Y/Z (mm)"), 2, 0)
        self.ob_x = _spin(-3000, 3000, 400); self.ob_y = _spin(-3000, 3000, 0)
        self.ob_z = _spin(-3000, 3000, 100)
        og.addWidget(self.ob_x, 2, 1); og.addWidget(self.ob_y, 2, 2)
        og.addWidget(self.ob_z, 2, 3)
        row = QHBoxLayout()
        for label, fn in (("Add box", self._add_obstacle),
                          ("Pedestal under robot", self._add_pedestal),
                          ("Duplicate", self._dup_obstacle),
                          ("Delete", self._del_obstacle)):
            b = QPushButton(label); b.clicked.connect(fn); row.addWidget(b)
        og.addLayout(row, 3, 0, 1, 4)
        root.addWidget(ob)

        # ---------- conveyors ----------
        cv = QGroupBox("Conveyors  (box source + obstacle)")
        cg = QGridLayout(cv)
        info = QLabel("A conveyor is a solid the arm avoids; its top surface is "
                      "where boxes arrive to be picked. ‘Set as pick source’ points "
                      "the palletizer’s pick at the belt top.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#5e81ac;font-size:10px")
        cg.addWidget(info, 0, 0, 1, 4)
        cg.addWidget(QLabel("Length×Width×Height (mm)"), 1, 0)
        self.cv_l = _spin(50, 8000, 800); self.cv_w = _spin(50, 5000, 350)
        self.cv_h = _spin(10, 3000, 400)
        self.cv_h.setToolTip("Belt-top height above the floor. The conveyor stands "
                             "on the floor; boxes are picked from its top.")
        cg.addWidget(self.cv_l, 1, 1); cg.addWidget(self.cv_w, 1, 2)
        cg.addWidget(self.cv_h, 1, 3)
        cg.addWidget(QLabel("Pos X/Y (mm) + yaw°"), 2, 0)
        self.cv_x = _spin(-3000, 3000, 300); self.cv_y = _spin(-3000, 3000, -450)
        self.cv_yaw = _spin(-180, 180, 0, dec=1, step=5)
        cg.addWidget(self.cv_x, 2, 1); cg.addWidget(self.cv_y, 2, 2)
        cg.addWidget(self.cv_yaw, 2, 3)
        crow = QHBoxLayout()
        for label, fn in (("Add conveyor", self._add_conveyor),
                          ("Set as pick source", self._set_pick_from_conveyor),
                          ("Delete", self._del_conveyor)):
            b = QPushButton(label); b.clicked.connect(fn); crow.addWidget(b)
        cg.addLayout(crow, 3, 0, 1, 4)
        root.addWidget(cv)

        # ---------- pallets ----------
        pl = QGroupBox("Pallets")
        pg = QGridLayout(pl)
        self.pallet_list = QListWidget()
        self.pallet_list.setMaximumHeight(80)
        self.pallet_list.currentRowChanged.connect(self._load_pallet_to_form)
        pg.addWidget(self.pallet_list, 0, 0, 1, 4)

        # standard pallet presets (ISO / GMA footprints)
        pg.addWidget(QLabel("Standard"), 1, 0)
        self.preset_box = QComboBox()
        self._pallet_presets = {
            "Custom": None,
            "EUR / EPAL 1  (1200×800×144)": (1200, 800, 144),
            "EUR 6 half   (800×600×144)": (800, 600, 144),
            "US GMA       (1219×1016×144)": (1219, 1016, 144),
            "Industrial   (1200×1000×144)": (1200, 1000, 144),
        }
        self.preset_box.addItems(list(self._pallet_presets.keys()))
        self.preset_box.setToolTip("Load a standard pallet footprint, then set "
                                   "box size / pattern / layers.")
        self.preset_box.currentTextChanged.connect(self._apply_pallet_preset)
        pg.addWidget(self.preset_box, 1, 1, 1, 3)

        pg.addWidget(QLabel("Pallet L×W×H (mm)"), 2, 0)
        self.pl_l = _spin(50, 5000, 800); self.pl_w = _spin(50, 5000, 600)
        self.pl_h = _spin(1, 2000, 144)
        pg.addWidget(self.pl_l, 2, 1); pg.addWidget(self.pl_w, 2, 2)
        pg.addWidget(self.pl_h, 2, 3)

        pg.addWidget(QLabel("Box L×W×H (mm) · kg"), 3, 0)
        self.bx_l = _spin(1, 2000, 200); self.bx_w = _spin(1, 2000, 150)
        self.bx_h = _spin(1, 2000, 120)
        self.bx_kg = _spin(0.0, 2000.0, 1.0, dec=1, step=0.5, suffix=" kg")
        self.bx_kg.setToolTip("Mass of one box — checked against the robot's "
                              "rated payload.")
        boxrow = QHBoxLayout()
        for w in (self.bx_l, self.bx_w, self.bx_h, self.bx_kg):
            boxrow.addWidget(w)
        pg.addLayout(boxrow, 3, 1, 1, 3)

        pg.addWidget(QLabel("Pattern / layers"), 4, 0)
        self.pattern_box = QComboBox()
        self.pattern_box.addItems(["column", "interlock", "brick"])
        self.pattern_box.setToolTip(
            "column = straight stack · interlock = alternate layers rotated 90° "
            "for a stable load · brick = alternate layers offset half a box.")
        pg.addWidget(self.pattern_box, 4, 1)
        self.layers = QSpinBox(); self.layers.setRange(1, 50); self.layers.setValue(3)
        pg.addWidget(QLabel("layers"), 4, 2); pg.addWidget(self.layers, 4, 3)

        pg.addWidget(QLabel("Gaps box/layer (mm)"), 5, 0)
        self.box_gap = _spin(0, 500, 5); self.layer_gap = _spin(0, 500, 0)
        pg.addWidget(self.box_gap, 5, 1); pg.addWidget(self.layer_gap, 5, 2)

        grip_lbl = QLabel("Grip/touch pt X/Y/Z (mm, box-local; blank=top)")
        grip_lbl.setToolTip("The point on each box where the tool touches/grips it, "
                            "measured from the box centre in the box's own frame. "
                            "Default is the top-face centre.")
        pg.addWidget(grip_lbl, 6, 0, 1, 2)
        self.grip_top = QPushButton("Top-centre")
        self.grip_top.setCheckable(True); self.grip_top.setChecked(True)
        self.grip_top.setToolTip("On = grip each box at its top-face centre. "
                                 "Off = set a custom touch point below.")
        self.grip_top.toggled.connect(self._grip_top_toggled)
        pg.addWidget(self.grip_top, 6, 2)
        self.gp_x = _spin(-1000, 1000, 0); self.gp_y = _spin(-1000, 1000, 0)
        self.gp_z = _spin(-1000, 1000, 60)
        gr = QHBoxLayout(); gr.addWidget(self.gp_x); gr.addWidget(self.gp_y)
        gr.addWidget(self.gp_z)
        pg.addLayout(gr, 7, 0, 1, 4)
        for s in (self.gp_x, self.gp_y, self.gp_z):
            s.setEnabled(False)

        pg.addWidget(QLabel("Pallet pos X/Y/Z (mm) + yaw°"), 8, 0)
        self.pp_x = _spin(-3000, 3000, 600); self.pp_y = _spin(-3000, 3000, 0)
        self.pp_z = _spin(-3000, 3000, 0); self.pp_yaw = _spin(-180, 180, 0, dec=1, step=5)
        pg.addWidget(self.pp_x, 8, 1); pg.addWidget(self.pp_y, 8, 2)
        pg.addWidget(self.pp_z, 8, 3)
        pg.addWidget(QLabel("yaw°"), 9, 0); pg.addWidget(self.pp_yaw, 9, 1)
        move_btn = QPushButton("Apply move to selected")
        move_btn.clicked.connect(self._apply_move)
        pg.addWidget(move_btn, 9, 2, 1, 2)

        pg.addWidget(QLabel("Role"), 10, 0)
        self.role_box = QComboBox()
        self.role_box.addItems(["Stack (palletize)", "Source (depalletize)",
                                "Destination"])
        self.role_box.setToolTip(
            "Stack: fill this pallet from the pick point / conveyor.\n"
            "Source: it starts full — the robot lifts boxes OFF it.\n"
            "Destination: the robot places the source's boxes ONTO it.\n"
            "Mark one Source + one Destination, then Simulate to depalletize "
            "from one pallet to the other.")
        self.role_box.currentIndexChanged.connect(self._role_changed)
        pg.addWidget(self.role_box, 10, 1, 1, 3)

        self.fit_lbl = QLabel("—")
        self.fit_lbl.setStyleSheet("color:#4c566a;font-size:10px")
        pg.addWidget(self.fit_lbl, 11, 0, 1, 4)
        for s in (self.pl_l, self.pl_w, self.bx_l, self.bx_w, self.box_gap):
            s.valueChanged.connect(self._update_fit)
        self.layers.valueChanged.connect(self._update_fit)
        self.pattern_box.currentTextChanged.connect(self._update_fit)

        starter = QPushButton("★ Add starter pallet fitted to this robot")
        starter.setStyleSheet("background:#a3be8c;font-weight:bold;padding:5px")
        starter.setToolTip("One click: adds a ready-to-run pallet sized and placed "
                           "to fit THIS robot's reach, so Simulate works immediately.")
        starter.clicked.connect(self._add_starter_pallet)
        pg.addWidget(starter, 12, 0, 1, 4)

        prow = QHBoxLayout()
        for label, fn in (("Add pallet", self._add_pallet),
                          ("Update selected", self._update_pallet),
                          ("Copy/Paste", self._copy_pallet),
                          ("Delete", self._del_pallet)):
            b = QPushButton(label); b.clicked.connect(fn); prow.addWidget(b)
        pg.addLayout(prow, 13, 0, 1, 4)
        root.addWidget(pl)

        # ---------- palletize / simulate ----------
        job = QGroupBox("Palletize  (pick → place)")
        jg = QGridLayout(job)
        pick_lbl = QLabel("Pick point X/Y/Z (mm)")
        pick_lbl.setToolTip("Where a box is presented to the robot (its top-centre). "
                            "The robot picks here, then places on the pallet.")
        jg.addWidget(pick_lbl, 0, 0)
        self.pick_x = _spin(-3000, 3000, -400); self.pick_y = _spin(-3000, 3000, -400)
        self.pick_z = _spin(-3000, 3000, 200)
        for s in (self.pick_x, self.pick_y, self.pick_z):
            s.setToolTip("Location where boxes arrive to be picked (base frame).")
        jg.addWidget(self.pick_x, 0, 1); jg.addWidget(self.pick_y, 0, 2)
        jg.addWidget(self.pick_z, 0, 3)
        appr_lbl = QLabel("Approach pick/place (mm)")
        appr_lbl.setToolTip("How far straight above the pick / place the tool starts "
                            "and ends each vertical move.")
        jg.addWidget(appr_lbl, 1, 0)
        self.appr_pick = _spin(0, 1000, 100); self.appr_place = _spin(0, 1000, 120)
        jg.addWidget(self.appr_pick, 1, 1); jg.addWidget(self.appr_place, 1, 2)
        sm_lbl = QLabel("Speed (m/s) / margin (mm)")
        sm_lbl.setToolTip("Margin = collision safety clearance added around every "
                          "link/box. Larger = more conservative (may flag near-misses).")
        jg.addWidget(sm_lbl, 2, 0)
        self.job_speed = _spin(0.001, 3.0, 0.25, dec=3, step=0.05)
        self.margin = _spin(0, 200, 20)
        self.margin.setToolTip("Collision safety clearance (mm) inflating every test.")
        jg.addWidget(self.job_speed, 2, 1); jg.addWidget(self.margin, 2, 2)

        # playback speed + fast planning (for quick iteration)
        pb = QHBoxLayout()
        pb.addWidget(QLabel("Sim speed"))
        self.speed_box = QComboBox()
        self._speed_map = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "4×": 4.0,
                           "8×": 8.0, "Instant": 0.0}
        self.speed_box.addItems(list(self._speed_map.keys()))
        self.speed_box.setCurrentText("2×")
        self.speed_box.setToolTip("Animation playback speed. 'Instant' jumps "
                                  "straight to the finished stack.")
        self.speed_box.currentTextChanged.connect(self._apply_speed)
        pb.addWidget(self.speed_box)
        self.fast_plan = QCheckBox("Fast plan")
        self.fast_plan.setToolTip("Coarser IK + sampling for much quicker "
                                  "planning while iterating. Re-run without it "
                                  "to confirm a job before exporting.")
        pb.addWidget(self.fast_plan)
        pb.addStretch(1)
        jg.addLayout(pb, 3, 0, 1, 4)

        brow = QHBoxLayout()
        for label, fn in (("Preview placements", self._preview),
                          ("Simulate palletization", self._simulate),
                          ("Add program → Program", self._to_program)):
            b = QPushButton(label); b.clicked.connect(fn); brow.addWidget(b)
        jg.addLayout(brow, 4, 0, 1, 4)
        self.report_lbl = QLabel("New here? Click ★ Add starter pallet, then "
                                 "Simulate palletization.")
        self.report_lbl.setWordWrap(True)
        self.report_lbl.setStyleSheet("font-weight:bold")
        jg.addWidget(self.report_lbl, 5, 0, 1, 4)
        root.addWidget(job)

        disc = QLabel("⚠ Feasibility aid only — nominal geometry, no cabling/"
                      "gripper/payload. Always dry-run on the real robot at "
                      "reduced speed with the E-stop in reach.")
        disc.setWordWrap(True)
        disc.setStyleSheet("color:#bf616a;font-size:10px")
        root.addWidget(disc)
        root.addStretch(1)
        self.apply_model_defaults()
        self._update_fit()
        self._apply_speed(self.speed_box.currentText())   # default 2× playback

    # ---- obstacle actions -------------------------------------------------
    def _add_obstacle(self) -> None:
        size = np.array([self.ob_l.value(), self.ob_w.value(), self.ob_h.value()]) / 1000
        c = np.array([self.ob_x.value(), self.ob_y.value(), self.ob_z.value()]) / 1000
        self.scene.add_obstacle(Box.from_size_center(size, c, name="obstacle"))
        self.report_lbl.setText("Added obstacle. The arm will avoid it in Simulate "
                                "and Run Offline Simulation.")

    def _add_pedestal(self) -> None:
        self.scene.add_pedestal(self.kin)
        self.report_lbl.setText("Added a pedestal under the robot base.")

    def _dup_obstacle(self) -> None:
        i = self.obs_list.currentRow()
        if i >= 0:
            self.scene.duplicate_obstacle(i)
            self.report_lbl.setText("Duplicated obstacle.")

    def _del_obstacle(self) -> None:
        i = self.obs_list.currentRow()
        if i >= 0:
            self.scene.remove_obstacle(i)
            self.report_lbl.setText("Deleted obstacle.")

    # ---- conveyor actions -------------------------------------------------
    def _add_conveyor(self) -> None:
        conv = self.scene.add_conveyor(
            self.cv_l.value() / 1000, self.cv_w.value() / 1000,
            self.cv_h.value() / 1000, x=self.cv_x.value() / 1000,
            y=self.cv_y.value() / 1000, yaw_deg=self.cv_yaw.value())
        # select it in the shared obstacle list and make it the pick source, so
        # one click gives a working box source the robot picks from.
        self.obs_list.setCurrentRow(len(self.scene.obstacles) - 1)
        self._set_pick_from_conveyor(conv)
        self.report_lbl.setText(
            f"Added conveyor '{conv.name}' — it's a collision obstacle and now the "
            f"pick source (boxes are picked from its top surface).")

    def _selected_conveyor(self):
        """The conveyor to act on: a selected/highlighted one if it is a conveyor,
        otherwise the most recently added conveyor (None if there are none)."""
        ref = self._sel_ref
        if ref and ref[0] == "obstacle" and 0 <= ref[1] < len(self.scene.obstacles):
            b = self.scene.obstacles[ref[1]]
            if b.kind == "conveyor":
                return b
        i = self.obs_list.currentRow()
        if 0 <= i < len(self.scene.obstacles) and self.scene.obstacles[i].kind == "conveyor":
            return self.scene.obstacles[i]
        convs = [b for b in self.scene.obstacles if b.kind == "conveyor"]
        return convs[-1] if convs else None

    def _set_pick_from_conveyor(self, conv=None) -> None:
        if conv is None or conv is False:              # button passes bool 'checked'
            conv = self._selected_conveyor()
        if conv is None:
            self.report_lbl.setText("Add or select a conveyor first, then set it as "
                                    "the pick source.")
            return
        bh = self.bx_h.value() / 1000.0                # grip the top of a box on it
        p = self.scene.conveyor_pick_point(conv, bh)
        self.pick_x.setValue(p[0] * 1000)
        self.pick_y.setValue(p[1] * 1000)
        self.pick_z.setValue(p[2] * 1000)
        self.report_lbl.setText(
            f"Pick source = '{conv.name}' top surface (X={p[0]*1000:.0f}, "
            f"Y={p[1]*1000:.0f}, Z={p[2]*1000:.0f} mm). Simulate to pick from it.")

    def _del_conveyor(self) -> None:
        conv = self._selected_conveyor()
        if conv is None:
            self.report_lbl.setText("No conveyor to delete.")
            return
        i = self.scene.obstacles.index(conv)
        self.scene.remove_obstacle(i)
        self.report_lbl.setText(f"Deleted conveyor '{conv.name}'.")

    # ---- pallet form ------------------------------------------------------
    def _grip_top_toggled(self, on: bool) -> None:
        for s in (self.gp_x, self.gp_y, self.gp_z):
            s.setEnabled(not on)

    def _apply_pallet_preset(self, text: str) -> None:
        dims = self._pallet_presets.get(text)
        if dims is None:
            return
        self.pl_l.setValue(dims[0]); self.pl_w.setValue(dims[1]); self.pl_h.setValue(dims[2])
        self.report_lbl.setText(f"Loaded {text.split('(')[0].strip()} footprint. "
                                f"Set box size / pattern / layers, then Simulate.")

    def _spec_from_form(self) -> PalletSpec:
        size = np.array([self.pl_l.value(), self.pl_w.value(), self.pl_h.value()]) / 1000
        box = np.array([self.bx_l.value(), self.bx_w.value(), self.bx_h.value()]) / 1000
        T = np.eye(4)
        a = np.radians(self.pp_yaw.value())
        T[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0],
                              [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        T[:3, 3] = np.array([self.pp_x.value(), self.pp_y.value(), self.pp_z.value()]) / 1000
        grip = (np.array([np.nan, np.nan, np.nan]) if self.grip_top.isChecked()
                else np.array([self.gp_x.value(), self.gp_y.value(),
                               self.gp_z.value()]) / 1000)
        return PalletSpec(name="Pallet", size=size, T=T, box_size=box,
                          box_weight_kg=self.bx_kg.value(),
                          layers=self.layers.value(),
                          box_gap=self.box_gap.value() / 1000,
                          layer_gap=self.layer_gap.value() / 1000,
                          grip_point=grip, pattern=self.pattern_box.currentText(),
                          role=_ROLES[self.role_box.currentIndex()])

    def _role_changed(self, idx: int) -> None:
        """Apply the role dropdown to the selected pallet immediately."""
        i = self.pallet_list.currentRow()
        if 0 <= i < len(self.scene.pallets):
            self.scene.pallets[i].role = _ROLES[int(idx)]
            self.scene.changed.emit()
            self.report_lbl.setText(
                f"'{self.scene.pallets[i].name}' role → {_ROLES[int(idx)]}.")

    def _load_pallet_to_form(self, i: int) -> None:
        if not (0 <= i < len(self.scene.pallets)):
            return
        s = self.scene.pallets[i]
        self.pl_l.setValue(s.size[0] * 1000); self.pl_w.setValue(s.size[1] * 1000)
        self.pl_h.setValue(s.size[2] * 1000)
        self.bx_l.setValue(s.box_size[0] * 1000); self.bx_w.setValue(s.box_size[1] * 1000)
        self.bx_h.setValue(s.box_size[2] * 1000)
        self.bx_kg.setValue(float(getattr(s, "box_weight_kg", 1.0)))
        pat = "column" if s.pattern in ("grid", "column") else s.pattern
        j = self.pattern_box.findText(pat)
        if j >= 0:
            self.pattern_box.setCurrentIndex(j)
        self.layers.setValue(s.layers)
        role = getattr(s, "role", "stack")
        self.role_box.blockSignals(True)               # loading, not user-editing
        self.role_box.setCurrentIndex(_ROLES.index(role) if role in _ROLES else 0)
        self.role_box.blockSignals(False)
        self.box_gap.setValue(s.box_gap * 1000); self.layer_gap.setValue(s.layer_gap * 1000)
        self.pp_x.setValue(s.T[0, 3] * 1000); self.pp_y.setValue(s.T[1, 3] * 1000)
        self.pp_z.setValue(s.T[2, 3] * 1000)
        self.pp_yaw.setValue(np.degrees(np.arctan2(s.T[1, 0], s.T[0, 0])))
        gp = np.asarray(s.grip_point, float)
        top = bool(np.any(np.isnan(gp)))
        self.grip_top.setChecked(top)
        if not top:
            self.gp_x.setValue(gp[0] * 1000); self.gp_y.setValue(gp[1] * 1000)
            self.gp_z.setValue(gp[2] * 1000)

    def _reach_defaults(self) -> dict:
        """Pallet/box/pick sizes and positions scaled to the current robot's reach
        so the out-of-the-box job is actually feasible."""
        try:
            r = float(self.kin.model.reach_mm) / 1000.0
        except Exception:                              # noqa: BLE001
            r = 0.85
        # Pallet placed well out in front (0.70·reach) so the elbow-up posture
        # clears the pallet edge on every arm — a floor-mounted robot stacking
        # onto a pallet right at its base grazes the near edge. These defaults
        # are verified feasible for every UR model.
        return dict(
            pallet=(0.36 * r, 0.28 * r, 0.12),
            box=(0.12 * r, 0.10 * r, 0.10 * r),
            pos=(0.70 * r, 0.0, 0.0),
            pick=(0.32 * r, -0.55 * r, 0.30 * r),
        )

    def apply_model_defaults(self) -> None:
        """Populate the form with reach-aware, feasible defaults for this robot."""
        d = self._reach_defaults()
        for spin, val in ((self.pl_l, d["pallet"][0]), (self.pl_w, d["pallet"][1]),
                          (self.pl_h, d["pallet"][2]), (self.bx_l, d["box"][0]),
                          (self.bx_w, d["box"][1]), (self.bx_h, d["box"][2]),
                          (self.pp_x, d["pos"][0]), (self.pp_y, d["pos"][1]),
                          (self.pp_z, d["pos"][2]), (self.pick_x, d["pick"][0]),
                          (self.pick_y, d["pick"][1]), (self.pick_z, d["pick"][2])):
            spin.setValue(val * 1000.0)
        # a realistic demo box mass that stays within the robot's payload
        try:
            payload = float(self.kin.model.payload_kg)
        except Exception:                              # noqa: BLE001
            payload = 5.0
        self.bx_kg.setValue(round(max(0.5, payload * 0.4), 1))

    def _add_starter_pallet(self) -> None:
        """One-click: reach-fitted pallet added and ready to Simulate."""
        self.apply_model_defaults()
        self.scene.add_pallet(self._spec_from_form())
        self.pallet_list.setCurrentRow(len(self.scene.pallets) - 1)
        name = self.scene.pallets[-1].name
        self.report_lbl.setText(
            f"Added '{name}' fitted to this robot — now click "
            f"“Simulate palletization”.")

    def _add_pallet(self) -> None:
        self.scene.add_pallet(self._spec_from_form())
        self.pallet_list.setCurrentRow(len(self.scene.pallets) - 1)
        self.report_lbl.setText(
            f"Added '{self.scene.pallets[-1].name}'. Preview or Simulate.")

    def _update_pallet(self) -> None:
        i = self.pallet_list.currentRow()
        if not (0 <= i < len(self.scene.pallets)):
            self.report_lbl.setText("Select a pallet in the list first.")
            return
        new = self._spec_from_form()
        new.name = self.scene.pallets[i].name
        self.scene.pallets[i] = new
        self.scene.changed.emit()
        self.report_lbl.setText(f"Updated '{new.name}'. Preview or Simulate.")

    def _copy_pallet(self) -> None:
        i = self.pallet_list.currentRow()
        if i < 0:
            self.report_lbl.setText("Select a pallet in the list to copy.")
            return
        w = float(self.scene.pallets[i].size[1])
        self.scene.duplicate_pallet(i, offset=(0.0, w + 0.1, 0.0))
        self.report_lbl.setText(
            f"Copied to '{self.scene.pallets[-1].name}' (beside the original). "
            f"Use Pallet pos + “Apply move” to reposition it.")

    def _del_pallet(self) -> None:
        i = self.pallet_list.currentRow()
        if i >= 0:
            name = self.scene.pallets[i].name
            self.scene.remove_pallet(i)
            self.report_lbl.setText(f"Deleted '{name}'.")

    def _apply_move(self) -> None:
        i = self.pallet_list.currentRow()
        if i < 0:
            self.report_lbl.setText("Select a pallet in the list to move.")
            return
        self.scene.move_pallet(i, x=self.pp_x.value() / 1000,
                               y=self.pp_y.value() / 1000,
                               z=self.pp_z.value() / 1000,
                               yaw_deg=self.pp_yaw.value())
        self.report_lbl.setText(
            f"Moved '{self.scene.pallets[i].name}' to X={self.pp_x.value():.0f} "
            f"Y={self.pp_y.value():.0f} Z={self.pp_z.value():.0f} mm.")

    def _update_fit(self) -> None:
        spec = self._spec_from_form()
        nx, ny = spec.grid_counts()
        total = spec.total_boxes()
        base = (f"Fits {nx}×{ny} = {nx*ny} boxes/base layer, "
                f"{total} total over {spec.layers} layer(s).")
        if spec.pattern == "interlock":
            base += "  Interlock rotates alternate layers 90°."
        elif spec.pattern == "brick":
            base += "  Brick offsets alternate layers half a box."
        self.fit_lbl.setText(base)

    # ---- palletize --------------------------------------------------------
    def _current_spec(self):
        i = self.pallet_list.currentRow()
        if 0 <= i < len(self.scene.pallets):
            return i, self.scene.pallets[i]
        if self.scene.pallets:
            return 0, self.scene.pallets[0]
        return -1, None

    def _apply_speed(self, text: str) -> None:
        self.main_window.set_sim_speed(self._speed_map.get(text, 1.0))

    def _job_opts(self) -> JobOptions:
        from robot.palletizer import _pose_down
        pick = _pose_down([self.pick_x.value() / 1000, self.pick_y.value() / 1000,
                           self.pick_z.value() / 1000])
        common = dict(pick_pose=pick,
                      pick_approach=self.appr_pick.value() / 1000,
                      place_approach=self.appr_place.value() / 1000,
                      speed_l=self.job_speed.value(),
                      margin=self.margin.value() / 1000)
        return (JobOptions.fast(**common) if self.fast_plan.isChecked()
                else JobOptions(**common))

    def _radii_cached(self):
        if self._radii is None:
            self._radii = link_radii(self.kin)
        return self._radii

    def _preview(self) -> None:
        if not self.scene.pallets:
            self.report_lbl.setText("No pallet yet — click ★ Add starter pallet.")
            return
        # preview every pallet's stack (each in its own colour)
        specs = []
        for p in self.scene.pallets:
            specs += [dict(T=pl.T, half=pl.half, color=p.color)
                      for pl in generate_placements(p)]
        self.viewport.build_placed_boxes(specs)
        self.viewport.set_placed_visible(len(specs))
        if specs:
            self.report_lbl.setText(
                f"Preview: {len(specs)} boxes across {len(self.scene.pallets)} "
                f"pallet(s). Click Simulate to run the robot.")
        else:
            self.report_lbl.setText("No boxes fit — the box is larger than the "
                                    "pallet. Increase pallet or reduce box size.")

    def _plan_sequence(self) -> dict:
        """Plan **every** pallet in list order as one continuous job. Each pallet
        is filled while the boxes already stacked on earlier pallets stand as
        obstacles, so the robot avoids them and errors out if it can't. The per-
        pallet sims/events/steps are chained into one timeline (box reveals get a
        global index; the animation stops at the first pallet that fails)."""
        opts = self._job_opts()
        q_prev = getattr(self.viewport, "_q", None)
        q_prev = (np.asarray(q_prev, float) if q_prev is not None
                  else np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]))
        g_sim = [q_prev.copy()]
        g_events: dict = {}
        g_steps = []
        placed_specs = []
        placed_offset = 0
        ok = True
        fail_cut = -1
        parts = []
        for k, spec in enumerate(self.scene.pallets):
            static = self.scene.static_boxes_for_sequence(k, set(range(k)))
            job = PalletJob(self.kin, static, spec, opts,
                            q_start=q_prev, radii=self._radii_cached())
            steps, sim, events, report = job.plan()
            sim = np.asarray(sim, float)
            pls = generate_placements(spec)
            placed_specs += [dict(T=p.T, half=p.half, color=spec.color) for p in pls]
            base = len(g_sim)                 # g_sim[-1] == this job's sim[0]
            for j, ev in events.items():
                if ev[0] == "drop_reveal":
                    ev = ("drop_reveal", placed_offset + ev[1])
                g_events[(base - 1) + j] = ev
            g_steps += steps
            g_sim.extend(sim[1:].tolist())
            placed_offset += len(pls)
            q_prev = np.asarray(g_sim[-1], float)
            parts.append(f"{spec.name}: {report.message}")
            if not report.ok:
                ok = False
                fail_cut = ((base - 1) + report.first_fail_sample + 1
                            if report.first_fail_sample >= 0 else len(g_sim))
                break
        return dict(steps=g_steps, sim=np.asarray(g_sim, float), events=g_events,
                    placed=placed_specs, ok=ok, fail_cut=fail_cut,
                    initial_visible=None,
                    message="   |   ".join(parts))

    def _has_transfer(self) -> bool:
        roles = [getattr(p, "role", "stack") for p in self.scene.pallets]
        return "source" in roles and "destination" in roles

    def _plan_transfer_scene(self) -> dict:
        """Plan a run that includes depalletize→palletize transfers. Source and
        destination pallets are paired in list order; any 'stack' pallets are
        palletized normally. Boxes on source pallets start visible and disappear
        as they're picked; destination/stack boxes appear as they're placed.
        Boxes present on other pallets stay as obstacles the robot avoids."""
        opts = self._job_opts()
        pallets = self.scene.pallets
        roles = [getattr(p, "role", "stack") for p in pallets]
        bases, base = [], 0
        for p in pallets:
            bases.append(base)
            base += p.total_boxes()
        placed_specs = []
        for p in pallets:
            placed_specs += [dict(T=pl.T, half=pl.half, color=p.color)
                             for pl in generate_placements(p)]
        initial_visible = [gi for i, p in enumerate(pallets) if roles[i] == "source"
                           for gi in range(bases[i], bases[i] + p.total_boxes())]
        srcs = [i for i, r in enumerate(roles) if r == "source"]
        dsts = [i for i, r in enumerate(roles) if r == "destination"]
        stacks = [i for i, r in enumerate(roles) if r == "stack"]
        present = set(srcs)                    # pallets currently holding boxes

        q_prev = getattr(self.viewport, "_q", None)
        q_prev = (np.asarray(q_prev, float) if q_prev is not None
                  else np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0]))
        g_sim = [q_prev.copy()]
        g_events: dict = {}
        g_steps = []
        parts, ok, fail_cut = [], True, -1

        def static_for(owned: set):
            boxes = [b for b in self.scene.obstacles if b.enabled]
            for j, p in enumerate(pallets):
                if j in owned:
                    continue
                boxes.append(p.pallet_box())
                if j in present:
                    boxes += [pl.to_box(f"{p.name}:box{k}")
                              for k, pl in enumerate(generate_placements(p))]
            return boxes

        def commit(steps, sim, events):
            sim = np.asarray(sim, float)
            base_i = len(g_sim)                # g_sim[-1] == this job's sim[0]
            for j, ev in events.items():
                g_events[(base_i - 1) + j] = ev
            g_steps.extend(steps)
            g_sim.extend(sim[1:].tolist())
            return base_i

        jobs = [("xfer", s, d) for s, d in zip(srcs, dsts)] + \
               [("stack", p_i, None) for p_i in stacks]
        for kind, a, b in jobs:
            if kind == "xfer":
                job = TransferJob(self.kin, static_for({a, b}), pallets[a], pallets[b],
                                  opts, q_start=q_prev, radii=self._radii_cached(),
                                  base_src=bases[a], base_dst=bases[b])
                steps, sim, events, report = job.plan()
                base_i = commit(steps, sim, events)
                present.discard(a); present.add(b)
                parts.append(f"{pallets[a].name}→{pallets[b].name}: {report.message}")
            else:
                job = PalletJob(self.kin, static_for({a}), pallets[a], opts,
                                q_start=q_prev, radii=self._radii_cached())
                steps, sim, events, report = job.plan()
                events = {j: (("box_show", bases[a] + ev[1] - 1)
                              if ev[0] == "drop_reveal" else ev)
                          for j, ev in events.items()}
                base_i = commit(steps, sim, events)
                present.add(a)
                parts.append(f"{pallets[a].name}: {report.message}")
            q_prev = np.asarray(g_sim[-1], float)
            if not report.ok:
                ok = False
                fail_cut = ((base_i - 1) + report.first_fail_sample + 1
                            if report.first_fail_sample >= 0 else len(g_sim))
                break
        return dict(steps=g_steps, sim=np.asarray(g_sim, float), events=g_events,
                    placed=placed_specs, ok=ok, fail_cut=fail_cut,
                    initial_visible=initial_visible,
                    message="   |   ".join(parts))

    def _cycle_estimate(self, sim, n_boxes: int) -> str:
        try:
            from robot.kinematics import TrajectoryPlanner
            if sim is None or len(sim) < 2 or n_boxes <= 0:
                return ""
            t = float(TrajectoryPlanner(self.kin).time_parameterize(sim)[-1])
            return f"  ~{t / n_boxes:.1f}s/box, ~{t:.0f}s total (est.)."
        except Exception:                              # noqa: BLE001
            return ""

    def _simulate(self) -> None:
        if not self.scene.pallets:
            self.report_lbl.setText("No pallet yet — click ★ Add starter pallet.")
            return
        self._apply_speed(self.speed_box.currentText())
        transfer = self._has_transfer()
        res = self._plan_transfer_scene() if transfer else self._plan_sequence()
        self.viewport.build_placed_boxes(res["placed"])
        init = res.get("initial_visible")
        if init is None:
            self.viewport.set_placed_visible(0)
        else:
            self.viewport.set_boxes_visible(init)
        sim, events = res["sim"], res["events"]
        # If infeasible, stop the animation right at the first failure so the
        # user sees exactly where and how it breaks.
        if not res["ok"] and res["fail_cut"] >= 0:
            cut = res["fail_cut"]
            sim = sim[:cut]
            events = {i: e for i, e in events.items() if i < cut}
        self.main_window.play_job(sim, events, initial_visible=init)
        total = sum(p.total_boxes() for p in self.scene.pallets)
        n = len(self.scene.pallets)
        scope = ("transfer" if transfer
                 else (f"{n} pallets" if n > 1 else "1 pallet"))
        if res["ok"]:
            tag = f"✓ FEASIBLE ({scope})"
            extra = self._cycle_estimate(res["sim"], total)
        elif total == 0:
            tag = "⚠ CHECK DIMENSIONS"
            extra = ""
        else:
            tag = "✗ STOPPED AT FAILURE"
            extra = ""
        self.report_lbl.setText(f"{tag} — {res['message']}{extra}")

    def _to_program(self) -> None:
        if not self.scene.pallets:
            self.report_lbl.setText("No pallet yet — click ★ Add starter pallet.")
            return
        res = (self._plan_transfer_scene() if self._has_transfer()
               else self._plan_sequence())
        if not res["steps"]:
            self.report_lbl.setText(
                "Nothing to add — no boxes fit on the pallet(s). Increase the "
                "pallet size or reduce the box/gaps.")
            return
        self.program_panel.add_program_steps(res["steps"])
        tag = "feasible ✓" if res["ok"] else "NOT feasible ✗ — review before running"
        self.report_lbl.setText(
            f"Added {len(res['steps'])} steps for {len(self.scene.pallets)} "
            f"pallet(s) to the Program ({tag}). {res['message']}")

    # ---- render -----------------------------------------------------------
    def _render_scene(self) -> None:
        # keep the selection valid if items were removed
        if self._sel_ref is not None:
            kind, i = self._sel_ref
            n = len(self.scene.obstacles) if kind == "obstacle" else len(self.scene.pallets)
            if not (0 <= i < n):
                self._sel_ref = None
        specs = self.scene.render_specs()
        for s in specs:
            if s.get("ref") == self._sel_ref:
                s["selected"] = True
        self.viewport.set_scene_boxes(specs)
        self.obs_list.clear()
        for b in self.scene.obstacles:
            sz = b.half * 2000
            self.obs_list.addItem(
                f"{b.name}  {sz[0]:.0f}×{sz[1]:.0f}×{sz[2]:.0f} mm")
        row = self.pallet_list.currentRow()
        self.pallet_list.clear()
        _role_tag = {"source": "  [SRC]", "destination": "  [DST]"}
        for p in self.scene.pallets:
            nx, ny = p.grid_counts()
            self.pallet_list.addItem(
                f"{p.name}  {nx*ny}×{p.layers}L  @({p.T[0,3]*1000:.0f},"
                f"{p.T[1,3]*1000:.0f}) mm{_role_tag.get(getattr(p, 'role', 'stack'), '')}")
        if 0 <= row < self.pallet_list.count():
            self.pallet_list.setCurrentRow(row)
        self._rebuild_axes()          # keep origin gizmos in step with edits

    # ---- origin-axis gizmos ----------------------------------------------
    def _rebuild_axes(self, *_) -> None:
        """(Re)draw the interactive origin gizmos for the toggled object types."""
        gizmos = []
        if self.ax_robot.isChecked():
            gizmos.append((("robot", 0), self.kin.base_pose(), 0.30))
        if self.ax_pallet.isChecked():
            for i, p in enumerate(self.scene.pallets):
                sc = float(np.clip(max(p.size[0], p.size[1]) * 0.5, 0.15, 0.40))
                gizmos.append((("pallet", i), p.T, sc))
        if self.ax_conveyor.isChecked():
            for i, b in enumerate(self.scene.obstacles):
                if b.kind == "conveyor":
                    sc = float(np.clip(max(b.half[0], b.half[1]), 0.15, 0.40))
                    gizmos.append((("obstacle", i), b.T, sc))
        self.viewport.set_gizmos(gizmos)

    def item_frame(self, ref) -> np.ndarray:
        """The full 4×4 world frame of a scene item (for the gizmo)."""
        kind, i = ref
        if kind == "robot":
            return self.kin.base_pose()
        if kind == "pallet" and 0 <= i < len(self.scene.pallets):
            return self.scene.pallets[i].T.copy()
        if kind == "obstacle" and 0 <= i < len(self.scene.obstacles):
            return self.scene.obstacles[i].T.copy()
        return np.eye(4)

    def set_item_pose(self, ref, T) -> None:
        """Live re-pose of an item from a gizmo drag (model + one actor, no
        full rebuild — keeps dragging smooth)."""
        kind, i = ref
        T = np.asarray(T, float)
        if kind == "robot":
            self.kin.set_base_pose(T)
            self.viewport.update_joints(self.viewport._q)
        elif kind == "pallet" and 0 <= i < len(self.scene.pallets):
            self.scene.pallets[i].T = T.copy()
            self.viewport.update_scene_actor(
                ("pallet", i), self.scene.pallets[i].pallet_box().T)
        elif kind == "obstacle" and 0 <= i < len(self.scene.obstacles):
            self.scene.obstacles[i].T = T.copy()
            self.viewport.update_scene_actor(("obstacle", i), T)

    def gizmo_commit(self, ref) -> None:
        """Finish a gizmo drag: refresh lists/collision (scene items) and redraw
        the gizmos at the final pose."""
        kind, _ = ref
        if kind == "robot":
            self._rebuild_axes()
            p = self.kin.base_pose()[:3, 3]
            self.report_lbl.setText(
                f"Moved robot base to X={p[0]*1000:.0f} Y={p[1]*1000:.0f} "
                f"Z={p[2]*1000:.0f} mm.")
        else:
            self.scene.changed.emit()          # refreshes lists + rebuilds axes
            self.report_lbl.setText(f"Moved {self._ref_name(ref)}.")

    # ---- direct 3D-scene editing (select / drag / delete / resize) --------
    def _install_scene_controller(self) -> None:
        try:
            iren = self.viewport.interactor()
            if iren is not None:
                self._scene_ctl = _SceneEditController(self, self.viewport, iren)
        except Exception:                              # noqa: BLE001
            self._scene_ctl = None

    def _ref_valid(self, ref) -> bool:
        if ref is None:
            return False
        kind, i = ref
        n = len(self.scene.obstacles) if kind == "obstacle" else len(self.scene.pallets)
        return 0 <= i < n

    def _ref_name(self, ref) -> str:
        if not self._ref_valid(ref):
            return "item"
        kind, i = ref
        return (self.scene.obstacles[i].name if kind == "obstacle"
                else self.scene.pallets[i].name)

    def item_position(self, ref) -> np.ndarray:
        if not self._ref_valid(ref):
            return np.zeros(3)
        kind, i = ref
        T = (self.scene.obstacles[i].T if kind == "obstacle"
             else self.scene.pallets[i].T)
        return T[:3, 3].copy()

    def select_ref(self, ref) -> None:
        if ref == self._sel_ref:
            return
        self._sel_ref = ref if self._ref_valid(ref) else None
        if self._sel_ref is not None:
            kind, i = self._sel_ref
            if kind == "obstacle" and i < self.obs_list.count():
                self.obs_list.setCurrentRow(i)
            elif kind == "pallet" and i < self.pallet_list.count():
                self.pallet_list.setCurrentRow(i)     # also loads the pallet form
            self.report_lbl.setText(
                f"Selected {self._ref_name(self._sel_ref)} — drag to move "
                f"(Shift = up/down), Del to delete, right-click for more.")
        self._render_scene()

    def drag_move(self, x=None, y=None, z=None) -> None:
        """Live re-pose during a drag — updates the model + just the one actor
        (no full rebuild / list refresh, so dragging stays smooth)."""
        ref = self._sel_ref
        if not self._ref_valid(ref):
            return
        kind, i = ref
        if kind == "obstacle":
            T = self.scene.obstacles[i].T.copy()
            if x is not None:
                T[0, 3] = x
            if y is not None:
                T[1, 3] = y
            if z is not None:
                T[2, 3] = z
            self.scene.obstacles[i].T = T
            self.viewport.update_scene_actor(ref, T)
        else:
            T = self.scene.pallets[i].T.copy()
            if x is not None:
                T[0, 3] = x
            if y is not None:
                T[1, 3] = y
            if z is not None:
                T[2, 3] = z
            self.scene.pallets[i].T = T
            self.viewport.update_scene_actor(ref, self.scene.pallets[i].pallet_box().T)

    def end_drag(self) -> None:
        """Commit a drag: one full refresh (lists + collision picking meta)."""
        p = self.item_position(self._sel_ref)
        self.scene.changed.emit()
        if self._ref_valid(self._sel_ref):
            self.report_lbl.setText(
                f"Moved {self._ref_name(self._sel_ref)} to X={p[0]*1000:.0f} "
                f"Y={p[1]*1000:.0f} Z={p[2]*1000:.0f} mm.")

    def delete_selected_scene(self) -> None:
        ref = self._sel_ref
        if not self._ref_valid(ref):
            return
        name = self._ref_name(ref)
        kind, i = ref
        self._sel_ref = None
        if kind == "obstacle":
            self.scene.remove_obstacle(i)
        else:
            self.scene.remove_pallet(i)
        self.report_lbl.setText(f"Deleted {name}.")

    def duplicate_selected_scene(self) -> None:
        ref = self._sel_ref
        if not self._ref_valid(ref):
            return
        kind, i = ref
        if kind == "obstacle":
            self.scene.duplicate_obstacle(i)
            self._sel_ref = ("obstacle", len(self.scene.obstacles) - 1)
        else:
            w = float(self.scene.pallets[i].size[1])
            self.scene.duplicate_pallet(i, offset=(0.0, w + 0.1, 0.0))
            self._sel_ref = ("pallet", len(self.scene.pallets) - 1)
        self._render_scene()
        self.report_lbl.setText(f"Duplicated to '{self._ref_name(self._sel_ref)}'.")

    def resize_selected_scene(self) -> None:
        ref = self._sel_ref
        if not self._ref_valid(ref):
            return
        kind, i = ref
        cur = (self.scene.obstacles[i].half * 2000.0 if kind == "obstacle"
               else self.scene.pallets[i].size * 1000.0)
        vals = []
        for axis, c in zip(("Length", "Width", "Height"), cur):
            v, ok = QInputDialog.getDouble(self, "Resize", f"{axis} (mm):",
                                           float(c), 1.0, 20000.0, 0)
            if not ok:
                return
            vals.append(v / 1000.0)
        if kind == "obstacle":
            self.scene.resize_obstacle(i, vals)
        else:
            self.scene.resize_pallet(i, vals)
        self.report_lbl.setText(f"Resized {self._ref_name(ref)} to "
                                f"{vals[0]*1000:.0f}×{vals[1]*1000:.0f}×"
                                f"{vals[2]*1000:.0f} mm.")

    def context_menu(self, ref) -> None:
        from PySide6.QtGui import QCursor
        menu = QMenu(self)
        menu.addAction("Duplicate", self.duplicate_selected_scene)
        menu.addAction("Resize…", self.resize_selected_scene)
        menu.addSeparator()
        menu.addAction("Delete", self.delete_selected_scene)
        menu.exec(QCursor.pos())


# ===========================================================================
#  Direct-manipulation controller — click/drag/delete scene objects
#
#  Attached to the *main* viewport's interactor, alongside the joint hand-guide
#  controller (which owns Ctrl/Shift + drag on a robot link). This one acts only
#  when a plain drag lands on a scene box:
#     * click a box         -> select it (highlight + syncs the panel)
#     * drag a box          -> move it on the ground plane (Shift = up/down)
#     * click empty space   -> clears selection, camera orbits as usual
#     * Delete / Backspace   -> delete the selected object
#     * right-click a box    -> Duplicate / Resize… / Delete menu
# ===========================================================================
class _SceneEditController:
    def __init__(self, panel: ScenePanel, viewport, iren):
        import vtk                                    # noqa: WPS433
        self._vtk = vtk
        self._p = panel
        self._vp = viewport
        self._iren = iren
        self._null = vtk.vtkInteractorStyleUser()
        self._saved = None
        self._moving = False
        self._giz = None                               # active gizmo-drag state
        self._mode = "xy"
        self._grab_off = np.zeros(2)
        self._z0 = 0.0
        self._plane_pt = np.zeros(3)
        self._plane_n = np.array([1.0, 0.0, 0.0])
        self._hit0_z = 0.0
        iren.AddObserver("LeftButtonPressEvent", self._press, 10.0)
        iren.AddObserver("MouseMoveEvent", self._move, 10.0)
        iren.AddObserver("LeftButtonReleaseEvent", self._release, 10.0)
        iren.AddObserver("RightButtonPressEvent", self._rclick, 10.0)
        iren.AddObserver("KeyPressEvent", self._key, 10.0)

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _ray_plane(p0, p1, pt, n):
        d = p1 - p0
        denom = float(np.dot(d, n))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(pt - p0, n) / denom)
        return p0 + t * d

    @staticmethod
    def _axis_param(p0, p1, ob, db):
        """Parameter s so ``ob + s·db`` is the point on the axis line closest to
        the view ray p0→p1 (used to drag an object along a single axis)."""
        d1 = p1 - p0
        w0 = p0 - ob
        a = float(np.dot(d1, d1)); b = float(np.dot(d1, db))
        c = float(np.dot(db, db)); d = float(np.dot(d1, w0)); e = float(np.dot(db, w0))
        den = a * c - b * b
        if abs(den) < 1e-9:
            return e / c if c > 1e-12 else None
        return (a * e - b * d) / den

    @staticmethod
    def _perp_basis(a):
        u = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(a, [0.0, 1.0, 0.0])
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(a, u)
        return u, v

    @staticmethod
    def _plane_angle(p, o, u, v):
        w = p - o
        return float(np.arctan2(float(np.dot(w, v)), float(np.dot(w, u))))

    @staticmethod
    def _axis_rot(a, ang):
        a = np.asarray(a, float)
        a = a / (np.linalg.norm(a) + 1e-12)
        x, y, z = a
        c, s = np.cos(ang), np.sin(ang)
        C = 1.0 - c
        return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                         [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                         [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])

    def _freeze(self):
        self._saved = self._iren.GetInteractorStyle()
        self._iren.SetInteractorStyle(self._null)

    def _thaw(self):
        if self._saved is not None:
            self._iren.SetInteractorStyle(self._saved)
            try:
                self._saved.StopState()
            except Exception:                          # noqa: BLE001
                pass
            self._saved = None

    # ---- gizmo drag -------------------------------------------------------
    def _begin_gizmo(self, g, x, y) -> None:
        ref, axis, kind = g
        T0 = self._p.item_frame(ref)
        o0 = T0[:3, 3].copy()
        a = T0[:3, axis].copy()
        a = a / (np.linalg.norm(a) + 1e-12)
        self._giz = dict(ref=ref, axis=axis, kind=kind, T0=T0.copy(), o0=o0, a=a)
        p0, p1 = self._vp.world_ray(x, y)
        if kind == "move":
            s0 = self._axis_param(p0, p1, o0, a)
            self._giz["s0"] = 0.0 if s0 is None else s0
        else:
            u, v = self._perp_basis(a)
            self._giz["u"], self._giz["v"] = u, v
            ph = self._ray_plane(p0, p1, o0, a)
            self._giz["ang0"] = self._plane_angle(ph, o0, u, v) if ph is not None else 0.0
        self._freeze()

    def _gizmo_move(self, x, y) -> None:
        g = self._giz
        p0, p1 = self._vp.world_ray(x, y)
        if g["kind"] == "move":
            s = self._axis_param(p0, p1, g["o0"], g["a"])
            if s is None:
                return
            new_o = g["o0"] + (s - g["s0"]) * g["a"]
            T = g["T0"].copy(); T[:3, 3] = new_o
            self._p.set_item_pose(g["ref"], T)
            M = np.eye(4); M[:3, 3] = new_o - g["o0"]
            self._vp.transform_gizmo(g["ref"], M)
        else:
            ph = self._ray_plane(p0, p1, g["o0"], g["a"])
            if ph is None:
                return
            d_ang = self._plane_angle(ph, g["o0"], g["u"], g["v"]) - g["ang0"]
            Rd = self._axis_rot(g["a"], d_ang)
            T = g["T0"].copy()
            T[:3, :3] = Rd @ g["T0"][:3, :3]
            T[:3, 3] = g["o0"]
            self._p.set_item_pose(g["ref"], T)
            M = np.eye(4); M[:3, :3] = Rd; M[:3, 3] = g["o0"] - Rd @ g["o0"]
            self._vp.transform_gizmo(g["ref"], M)

    # ---- observers --------------------------------------------------------
    def _press(self, obj, evt):
        x, y = self._iren.GetEventPosition()
        g = self._vp.gizmo_pick(x, y)
        if g is not None:
            self._begin_gizmo(g, x, y)             # drag an origin-axis handle
            return
        ref = self._vp.scene_pick(x, y)
        if ref is None:
            if self._p._sel_ref is not None:
                self._p.select_ref(None)               # click empty = deselect
            return                                     # let the camera orbit
        self._p.select_ref(ref)
        pos = self._p.item_position(ref)
        self._z0 = float(pos[2])
        self._mode = "z" if self._iren.GetShiftKey() else "xy"
        p0, p1 = self._vp.world_ray(x, y)
        if self._mode == "xy":
            hit = self._ray_plane(p0, p1, np.array([0, 0, self._z0]),
                                  np.array([0.0, 0.0, 1.0]))
            self._grab_off = (pos[:2] - hit[:2]) if hit is not None else np.zeros(2)
        else:
            self._plane_pt = pos.copy()
            self._plane_n = self._vp.camera_horizontal_normal()
            hit = self._ray_plane(p0, p1, self._plane_pt, self._plane_n)
            self._hit0_z = float(hit[2]) if hit is not None else self._z0
        self._moving = True
        self._freeze()

    def _move(self, obj, evt):
        if self._giz is not None:
            x, y = self._iren.GetEventPosition()
            self._gizmo_move(x, y)
            return
        if not self._moving:
            return
        x, y = self._iren.GetEventPosition()
        p0, p1 = self._vp.world_ray(x, y)
        if self._mode == "xy":
            hit = self._ray_plane(p0, p1, np.array([0, 0, self._z0]),
                                  np.array([0.0, 0.0, 1.0]))
            if hit is None:
                return
            self._p.drag_move(x=float(hit[0] + self._grab_off[0]),
                              y=float(hit[1] + self._grab_off[1]))
        else:
            hit = self._ray_plane(p0, p1, self._plane_pt, self._plane_n)
            if hit is None:
                return
            self._p.drag_move(z=float(self._z0 + (hit[2] - self._hit0_z)))

    def _release(self, obj, evt):
        if self._giz is not None:
            ref = self._giz["ref"]
            self._giz = None
            self._thaw()
            self._p.gizmo_commit(ref)
            return
        if not self._moving:
            return
        self._moving = False
        self._thaw()
        self._p.end_drag()

    def _rclick(self, obj, evt):
        x, y = self._iren.GetEventPosition()
        ref = self._vp.scene_pick(x, y)
        if ref is None:
            return                                     # empty space → camera zoom
        self._p.select_ref(ref)
        self._freeze()                                 # don't let the style dolly
        try:
            self._p.context_menu(ref)
        finally:
            self._thaw()

    def _key(self, obj, evt):
        try:
            k = self._iren.GetKeySym()
        except Exception:                              # noqa: BLE001
            return
        if k in ("Delete", "BackSpace") and self._p._sel_ref is not None:
            self._p.delete_selected_scene()
