# UR GUI Simulator

A full-featured desktop application (PySide6 / Qt) to program, simulate, and
control any Universal Robot — **UR3 · UR5 · UR10 · UR16 · UR20 · UR30**, in both
**CB3** and **e-Series** variants.

It combines a live digital twin, a teach pendant, a visual program builder,
CAD-driven toolpath generation, and dual-mode (URScript ↔ Python) code
generation in one window.

---

## Quick start

```bash
python -m pip install -r requirements.txt
python main.py
```

Only **PySide6 + numpy** are strictly required — the app launches and runs the
digital twin / offline simulator with nothing else installed. Every heavier
dependency unlocks a feature and degrades gracefully when absent.

```bash
python main.py --ip 192.168.1.10 --model UR10e --connect --log DEBUG
```

| Flag | Meaning |
|------|---------|
| `--ip` | Robot IP (default `192.168.1.100`) |
| `--model` | UR3 · UR3e · UR5 · UR5e · UR10 · UR10e · UR16e · UR20e · UR30e |
| `--connect` | Attempt connection on startup |
| `--log` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Architecture

```
main.py                     App entry: logging, excepthook, window loader
robot/
  ur_models.py              DH / limits / payload registry for all UR arms
  ur_bridge.py              Threaded RTDE + socket + simulation bridge
  kinematics.py             FK / IK (damped least squares) / trajectory planner
  program.py                Visual program model + URScript & Python generators
cad/
  cad_importer.py           STEP/IGES/STL/DXF loader, 3-pt frame calibration, toolpaths
gui/
  viewport_3d.py            PyVista/VTK digital-twin canvas embedded in Qt
ui/
  main_window.py            Dockable-widget shell + render/animation loop
  panels/
    connection_panel.py     IP/ports/model, status, dashboard, E-Stop, freedrive
    jog_panel.py            Joint + Cartesian jogging, teach waypoint
    program_panel.py        Reorderable program tree, simulate, execute
    editor_panel.py         URScript / Python tabs, syntax highlighting, export
    cad_panel.py            Import, calibrate, generate & attach toolpaths
```

### Threading model
All robot I/O runs on a background `QThread` (`URWorker`). Robot→GUI state flows
**only** through the `state_updated` Qt signal; GUI→robot commands are queued
onto the worker thread via `QMetaObject.invokeMethod`. No widget is ever touched
off the GUI thread, and no socket is ever touched on it — so the UI never
freezes, even at a 125 Hz feedback rate.

### Communication backends (auto-selected)
1. **RTDE** via `ur_rtde` — calibrated feedback + clean control API + dashboard.
2. **Raw URScript socket** (port 30002) + dashboard socket (29999) fallback.
3. **Pure simulation** — a virtual, rate-limited joint model so the whole UI
   (jogging, twin, program playback, code-gen) works fully offline.

---

## Feature status

| Module | Status |
|--------|--------|
| Connection / model / status / dashboard / E-Stop / freedrive | ✅ |
| Joint + Cartesian jogging (Base/Tool frame), teach waypoints | ✅ |
| Program builder: MoveJ/L/P, Process, gripper, delay, DO, wait DI | ✅ |
| Reorder / edit / enable / delete / save / load (`.urgui`) | ✅ |
| FK/IK/trajectory for all 6 arms (verified sub-mm) | ✅ |
| 3D digital twin (live + offline animation) | ✅ (needs a display/GPU) |
| CAD import STL/OBJ/PLY (trimesh) · DXF (ezdxf) | ✅ |
| CAD import STEP/IGES | ⚙ needs `pythonocc-core` (conda) |
| 3-point frame calibration · toolpath `T_base_tool = T_base_cad·T_cad_path` | ✅ verified |
| URScript + Python (`ur_rtde`) generation / export / execute | ✅ |

---

## Optional dependencies

| Want | Install |
|------|---------|
| RTDE control & feedback | `pip install ur_rtde` |
| 3D viewport | `pip install pyvista pyvistaqt vtk` |
| Mesh + DXF CAD | `pip install trimesh ezdxf scipy` |
| STEP / IGES CAD | `conda install -c conda-forge pythonocc-core` |
| URDF-based IK (alt) | `pip install ikpy` |

---

## Safety

* **Execute on Robot** and **Execute URScript** always prompt for confirmation.
* The E-Stop button (and the `Esc` shortcut) issue a protective stop through the
  control API, the dashboard, *and* freeze the virtual target — whichever
  backend is live.
* Always keep a hardware E-Stop within reach when driving a physical robot.

---

## Notes

* Nominal DH parameters are used; a connected controller's factory calibration
  deltas (exposed over RTDE) can be layered on top for higher accuracy.
* The 3D viewport requires a real OpenGL context — it will not render in a
  headless CI environment, but all non-visual logic is fully testable there.
