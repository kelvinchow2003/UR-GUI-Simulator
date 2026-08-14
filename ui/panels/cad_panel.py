"""
ui/panels/cad_panel.py
==================================================================
CAD Import & Relative Trajectory Mapping dock.

    * Import .step/.iges/.stl/.dxf  (backend-gated)
    * 3-point frame calibration: origin / +X / XY-plane  ->  T_base_cad
    * Toolpath generation from CAD edges / polylines, offset along the
      surface normal, mapped to the base frame:  T_base_tool = T_base_cad · T_cad_path
    * Push the generated toolpath into the Program as Process points
    * Everything is echoed to the 3D viewport (mesh, frame triad, path)
==================================================================
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QMessageBox, QDoubleSpinBox, QComboBox, QLineEdit,
    QFormLayout,
)

from cad.cad_importer import (
    CADImporter, CADImportError, FrameCalibrator, ToolpathGenerator, LoadedCAD,
)


class CADPanel(QWidget):
    cad_loaded = Signal(object)          # LoadedCAD
    frame_changed = Signal(object)       # 4x4 np.ndarray
    toolpath_ready = Signal(object)      # (N,6) poses

    def __init__(self, program_panel, viewport, parent=None):
        super().__init__(parent)
        self.program_panel = program_panel
        self.viewport = viewport
        self.importer = CADImporter()
        self.calibrator = FrameCalibrator()
        self.pathgen = ToolpathGenerator()
        self.cad: LoadedCAD | None = None
        self.T_base_cad = np.eye(4)
        self.toolpath = None
        self._build()

    # ---- UI ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        # import
        imp = QGroupBox("Import")
        il = QVBoxLayout(imp)
        self.import_btn = QPushButton("Import CAD file…")
        self.import_btn.clicked.connect(self._import)
        il.addWidget(self.import_btn)
        fmts = ", ".join(self.importer.supported_formats()) or "none (install backends)"
        self.info_lbl = QLabel(f"Available: {fmts}")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color:#888;font-size:10px")
        il.addWidget(self.info_lbl)
        self.loaded_lbl = QLabel("No CAD loaded.")
        self.loaded_lbl.setWordWrap(True)
        il.addWidget(self.loaded_lbl)
        root.addWidget(imp)

        # calibration
        cal = QGroupBox("3-point frame calibration  (metres, base frame)")
        cf = QFormLayout(cal)
        self.origin_edit = QLineEdit("0, 0, 0")
        self.xpt_edit = QLineEdit("0.1, 0, 0")
        self.ypt_edit = QLineEdit("0, 0.1, 0")
        cf.addRow("Origin", self.origin_edit)
        cf.addRow("+X point", self.xpt_edit)
        cf.addRow("XY-plane point", self.ypt_edit)
        hb = QHBoxLayout()
        pick = QPushButton("Compute T_base_cad")
        pick.clicked.connect(self._compute_frame)
        ident = QPushButton("Reset to identity")
        ident.clicked.connect(self._reset_frame)
        hb.addWidget(pick); hb.addWidget(ident)
        cf.addRow(hb)
        self.frame_lbl = QLabel("T_base_cad = identity")
        self.frame_lbl.setStyleSheet("font-family:Consolas;font-size:10px")
        cf.addRow(self.frame_lbl)
        root.addWidget(cal)

        # toolpath
        tp = QGroupBox("Toolpath generation")
        tg = QGridLayout(tp)
        tg.addWidget(QLabel("Approach"), 0, 0)
        self.approach_box = QComboBox()
        self.approach_box.addItems(["-Z (down)", "+Z (up)", "-Y", "+X"])
        tg.addWidget(self.approach_box, 0, 1)
        tg.addWidget(QLabel("Offset (mm)"), 1, 0)
        self.offset = QDoubleSpinBox(); self.offset.setRange(-100, 100)
        self.offset.setValue(0.0)
        tg.addWidget(self.offset, 1, 1)
        tg.addWidget(QLabel("Speed (m/s)"), 2, 0)
        self.tp_speed = QDoubleSpinBox(); self.tp_speed.setRange(0.001, 1.0)
        self.tp_speed.setValue(0.1); self.tp_speed.setDecimals(3)
        tg.addWidget(self.tp_speed, 2, 1)

        gen = QPushButton("Generate toolpath from CAD")
        gen.clicked.connect(self._generate)
        add = QPushButton("Add toolpath → Program")
        add.clicked.connect(self._add_to_program)
        tg.addWidget(gen, 3, 0, 1, 2)
        tg.addWidget(add, 4, 0, 1, 2)
        self.tp_lbl = QLabel("No toolpath.")
        tg.addWidget(self.tp_lbl, 5, 0, 1, 2)
        root.addWidget(tp)
        root.addStretch(1)

    # ---- import -----------------------------------------------------------
    def _import(self) -> None:
        exts = " ".join(f"*{e}" for e in self.importer.supported_formats())
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CAD", "", f"CAD files ({exts});;All files (*)")
        if not path:
            return
        try:
            self.cad = self.importer.load(path)
        except CADImportError as exc:
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        lo, hi = self.cad.bounds
        size = (hi - lo) * 1000
        kind = "mesh" if self.cad.is_mesh else f"{len(self.cad.polylines)} curve(s)"
        self.loaded_lbl.setText(
            f"Loaded: {path.split('/')[-1]}\n{kind}, "
            f"{len(self.cad.vertices)} verts\n"
            f"bbox ≈ {size[0]:.0f} × {size[1]:.0f} × {size[2]:.0f} mm")
        self.viewport.show_cad(self.cad)
        self.cad_loaded.emit(self.cad)

    # ---- calibration ------------------------------------------------------
    @staticmethod
    def _parse_vec(text: str) -> np.ndarray:
        parts = [p for p in text.replace(",", " ").split() if p]
        return np.array([float(x) for x in parts[:3]])

    def _compute_frame(self) -> None:
        try:
            o = self._parse_vec(self.origin_edit.text())
            x = self._parse_vec(self.xpt_edit.text())
            y = self._parse_vec(self.ypt_edit.text())
            self.T_base_cad = self.calibrator.from_three_points(o, x, y)
        except (ValueError, IndexError) as exc:
            QMessageBox.warning(self, "Calibration error",
                                f"Could not build a frame:\n{exc}")
            return
        self.pathgen.set_frame(self.T_base_cad)
        p = self.T_base_cad[:3, 3]
        self.frame_lbl.setText(
            f"origin = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]\n"
            "R set from picked axes")
        self.viewport.show_frame(self.T_base_cad)
        self.frame_changed.emit(self.T_base_cad)

    def _reset_frame(self) -> None:
        self.T_base_cad = np.eye(4)
        self.pathgen.set_frame(self.T_base_cad)
        self.frame_lbl.setText("T_base_cad = identity")
        self.viewport.show_frame(self.T_base_cad)
        self.frame_changed.emit(self.T_base_cad)

    # ---- toolpath ---------------------------------------------------------
    def _generate(self) -> None:
        if self.cad is None:
            QMessageBox.information(self, "Toolpath", "Import a CAD file first.")
            return
        approach = {
            "-Z (down)": (0, 0, -1), "+Z (up)": (0, 0, 1),
            "-Y": (0, -1, 0), "+X": (1, 0, 0),
        }[self.approach_box.currentText()]
        offset = self.offset.value() / 1000.0
        paths = self.pathgen.from_edges(self.cad, approach=approach, offset_m=offset)
        if not paths:
            QMessageBox.information(self, "Toolpath",
                                    "No extractable edges/curves found in CAD.")
            return
        # concatenate all path segments into one pose array
        poses = np.vstack([p.poses for p in paths if len(p)])
        self.toolpath = poses
        self.tp_lbl.setText(f"{len(poses)} points across {len(paths)} segment(s).")
        self.viewport.show_toolpath(poses)
        self.toolpath_ready.emit(poses)

    def _add_to_program(self) -> None:
        if self.toolpath is None:
            QMessageBox.information(self, "Toolpath", "Generate a toolpath first.")
            return
        self.program_panel.add_toolpath(self.toolpath, speed=self.tp_speed.value())
