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
    PalletSpec, JobOptions, PalletJob, generate_placements,
)


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

        self.fit_lbl = QLabel("—")
        self.fit_lbl.setStyleSheet("color:#4c566a;font-size:10px")
        pg.addWidget(self.fit_lbl, 10, 0, 1, 4)
        for s in (self.pl_l, self.pl_w, self.bx_l, self.bx_w, self.box_gap):
            s.valueChanged.connect(self._update_fit)
        self.layers.valueChanged.connect(self._update_fit)
        self.pattern_box.currentTextChanged.connect(self._update_fit)

        starter = QPushButton("★ Add starter pallet fitted to this robot")
        starter.setStyleSheet("background:#a3be8c;font-weight:bold;padding:5px")
        starter.setToolTip("One click: adds a ready-to-run pallet sized and placed "
                           "to fit THIS robot's reach, so Simulate works immediately.")
        starter.clicked.connect(self._add_starter_pallet)
        pg.addWidget(starter, 11, 0, 1, 4)

        prow = QHBoxLayout()
        for label, fn in (("Add pallet", self._add_pallet),
                          ("Update selected", self._update_pallet),
                          ("Copy/Paste", self._copy_pallet),
                          ("Delete", self._del_pallet)):
            b = QPushButton(label); b.clicked.connect(fn); prow.addWidget(b)
        pg.addLayout(prow, 12, 0, 1, 4)
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
                          grip_point=grip, pattern=self.pattern_box.currentText())

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
        i, spec = self._current_spec()
        if spec is None:
            self.report_lbl.setText("No pallet yet — click ★ Add starter pallet.")
            return
        placements = generate_placements(spec)
        specs = [dict(T=p.T, half=p.half) for p in placements]
        self.viewport.build_placed_boxes(specs, color=spec.color)
        self.viewport.set_placed_visible(len(specs))
        if specs:
            self.report_lbl.setText(f"Preview: {len(specs)} boxes on '{spec.name}'. "
                                    f"Click Simulate to test the robot.")
        else:
            self.report_lbl.setText("No boxes fit — the box is larger than the "
                                    "pallet. Increase pallet or reduce box size.")

    def _build_job(self):
        i, spec = self._current_spec()
        if spec is None:
            self.report_lbl.setText("No pallet yet — click ★ Add starter pallet.")
            return None, None
        q0 = getattr(self.viewport, "_q", None)
        static = self.scene.static_boxes(exclude_pallet=i)
        job = PalletJob(self.kin, static, spec, self._job_opts(),
                        q_start=q0, radii=self._radii_cached())
        return job, spec

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
        job, spec = self._build_job()
        if job is None:
            return
        self._apply_speed(self.speed_box.currentText())
        steps, sim, events, report = job.plan()
        placements = generate_placements(spec)
        self.viewport.build_placed_boxes(
            [dict(T=p.T, half=p.half) for p in placements], color=spec.color)
        self.viewport.set_placed_visible(0)
        # If infeasible, stop the animation right at the first failure so the
        # user sees exactly where and how it breaks.
        if not report.ok and report.first_fail_sample >= 0:
            cut = report.first_fail_sample + 1
            sim = sim[:cut]
            events = {i: e for i, e in events.items() if i < cut}
        # hand the job to the main window's render loop to animate
        self.main_window.play_job(sim, events)
        if report.ok:
            tag = "✓ FEASIBLE"
            extra = self._cycle_estimate(sim, spec.total_boxes())
        elif spec.total_boxes() == 0:
            tag = "⚠ CHECK DIMENSIONS"
            extra = ""
        else:
            tag = "✗ STOPPED AT FAILURE"
            extra = ""
        self.report_lbl.setText(f"{tag} — {report.message}{extra}")

    def _to_program(self) -> None:
        job, spec = self._build_job()
        if job is None:
            return
        steps, sim, events, report = job.plan()
        if not steps:
            self.report_lbl.setText(
                "Nothing to add — no boxes fit on the pallet. Increase the pallet "
                "size or reduce the box/gaps.")
            return
        self.program_panel.add_program_steps(steps)
        tag = "feasible ✓" if report.ok else "NOT feasible ✗ — review before running"
        self.report_lbl.setText(
            f"Added {len(steps)} steps for '{spec.name}' to the Program ({tag}). "
            f"{report.message}")

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
        for p in self.scene.pallets:
            nx, ny = p.grid_counts()
            self.pallet_list.addItem(
                f"{p.name}  {nx*ny}×{p.layers}L  @({p.T[0,3]*1000:.0f},"
                f"{p.T[1,3]*1000:.0f}) mm")
        if 0 <= row < self.pallet_list.count():
            self.pallet_list.setCurrentRow(row)

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

    # ---- observers --------------------------------------------------------
    def _press(self, obj, evt):
        x, y = self._iren.GetEventPosition()
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
