"""
mcp_server.py
==================================================================
An **MCP server** that lets Claude Desktop build a simulation in the UR GUI
Simulator from natural-language instructions ("use a UR30, two pallets 10 cm
apart, palletize them").

How it works
------------
The server keeps an in-memory session (robot model + obstacles + pallets +
program) built from the app's own model classes. Each tool call updates that
session and writes a **project file** (``.urgproj``) to a fixed *live* path.
The running app watches that file and auto-reloads, so the cell appears and
updates in the app as Claude works. It also works when the app is closed —
``launch_app`` starts it pointed at the live file.

The natural-language understanding happens in Claude Desktop; this server just
exposes clean, structured tools it can call.

Units
-----
Tool arguments use **millimetres** and **degrees** (matching the app's Scene
panel), so instructions like "10 cm away" map to ``x_mm=100``. Internally
everything is converted to the app's base-frame metres.

Setup (Claude Desktop)
----------------------
Install the SDK into the project venv::

    .venv/Scripts/python -m pip install "mcp[cli]"

Add to ``claude_desktop_config.json`` (Settings ▸ Developer ▸ Edit Config)::

    {
      "mcpServers": {
        "ur-gui": {
          "command": "C:\\\\Users\\\\kelvi\\\\Desktop\\\\UR GUI Sim\\\\UR-GUI-Simulator\\\\.venv\\\\Scripts\\\\python.exe",
          "args": ["C:\\\\Users\\\\kelvi\\\\Desktop\\\\UR GUI Sim\\\\UR-GUI-Simulator\\\\mcp_server.py"]
        }
      }
    }

Restart Claude Desktop; the "ur-gui" tools appear. Run this file directly to
self-test the session logic without Claude::

    .venv/Scripts/python mcp_server.py --selftest
==================================================================
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

from robot.kinematics import Kinematics
from robot.collision import Box
from robot.palletizer import PalletSpec, JobOptions
from robot.program import Program
from robot.scene_planner import plan_scene
from robot.project import project_dict
from robot.ur_models import UR_MODELS, MODEL_NAMES
import json

# The app watches this file and live-reloads it. Override with UR_GUI_LIVE_FILE.
LIVE_FILE = os.environ.get(
    "UR_GUI_LIVE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_session.urgproj"))

_MM = 1000.0


def resolve_model(name: str) -> str:
    """Map a loose model name to a known one: 'UR30' → 'UR30e' (the app only
    ships the e-Series for the big arms), case-insensitive."""
    if name in UR_MODELS:
        return name
    for cand in (name, name + "e", name.upper(), name.upper() + "e"):
        if cand in UR_MODELS:
            return cand
    raise ValueError(f"Unknown robot model '{name}'. Known: {', '.join(MODEL_NAMES)}")


# ===========================================================================
#  Session — the testable core (no MCP dependency)
# ===========================================================================
class AppSession:
    """Holds the scene being built and writes it to the live project file."""

    def __init__(self, live_file: str = LIVE_FILE):
        self.live_file = live_file
        self.model_name = "UR10e"
        self.obstacles: List[Box] = []
        self.pallets: List[PalletSpec] = []
        self.program = Program(name="MCP Session")

    # ---- geometry helpers -------------------------------------------------
    def _pedestal_height(self) -> float:
        tops = [float(b.T[2, 3] + b.half[2]) for b in self.obstacles
                if b.kind == "pedestal" and b.enabled]
        return max(tops) if tops else 0.0

    def _base_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[2, 3] = self._pedestal_height()               # stand on any pedestal
        return T

    def _unique(self, name: str, existing: List[str]) -> str:
        if name not in existing:
            return name
        i = 2
        while f"{name} {i}" in existing:
            i += 1
        return f"{name} {i}"

    def kin(self) -> Kinematics:
        k = Kinematics(UR_MODELS[self.model_name])
        k.set_base_height(self._pedestal_height())
        return k

    # ---- mutating operations ---------------------------------------------
    def set_model(self, name: str) -> str:
        self.model_name = resolve_model(name)
        return self.model_name

    def add_pallet(self, x_mm=600.0, y_mm=0.0, yaw_deg=0.0,
                   length_mm=800.0, width_mm=600.0, slab_mm=144.0,
                   box_l_mm=200.0, box_w_mm=150.0, box_h_mm=120.0,
                   box_weight_kg=1.0, layers=3, pattern="column",
                   role="stack", name: Optional[str] = None) -> str:
        a = np.radians(yaw_deg)
        T = np.eye(4)
        T[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0],
                              [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        T[:3, 3] = [x_mm / _MM, y_mm / _MM, 0.0]
        nm = self._unique(name or f"Pallet {len(self.pallets) + 1}",
                          [p.name for p in self.pallets])
        spec = PalletSpec(
            name=nm, size=np.array([length_mm, width_mm, slab_mm]) / _MM, T=T,
            box_size=np.array([box_l_mm, box_w_mm, box_h_mm]) / _MM,
            box_weight_kg=float(box_weight_kg), layers=int(layers),
            pattern=pattern, role=role)
        self.pallets.append(spec)
        return nm

    def add_conveyor(self, x_mm=400.0, y_mm=-400.0, yaw_deg=0.0,
                     length_mm=800.0, width_mm=300.0, height_mm=200.0) -> str:
        a = np.radians(yaw_deg)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                      [0, 0, 1]])
        size = np.array([length_mm, width_mm, height_mm]) / _MM
        centre = [x_mm / _MM, y_mm / _MM, height_mm / _MM / 2.0]
        nm = self._unique("conveyor", [b.name for b in self.obstacles])
        self.obstacles.append(Box.from_size_center(size, centre, name=nm,
                                                    kind="conveyor", R=R))
        return nm

    def add_obstacle(self, x_mm, y_mm, z_mm, length_mm, width_mm, height_mm,
                     kind="obstacle", name: Optional[str] = None) -> str:
        size = np.array([length_mm, width_mm, height_mm]) / _MM
        centre = [x_mm / _MM, y_mm / _MM, z_mm / _MM]
        nm = self._unique(name or kind, [b.name for b in self.obstacles])
        self.obstacles.append(Box.from_size_center(size, centre, name=nm, kind=kind))
        return nm

    def add_pedestal(self, height_mm=300.0, side_mm=400.0) -> str:
        """A pedestal directly under the base that mounts the robot higher."""
        z0 = self._pedestal_height()
        size = np.array([side_mm, side_mm, height_mm]) / _MM
        centre = [0.0, 0.0, z0 + size[2] / 2.0]
        nm = self._unique("pedestal", [b.name for b in self.obstacles])
        self.obstacles.append(Box.from_size_center(size, centre, name=nm,
                                                    kind="pedestal"))
        return nm

    def clear(self) -> None:
        self.obstacles = []
        self.pallets = []
        self.program = Program(name="MCP Session")

    def plan(self, smart_posture=True, fast=False) -> "object":
        """Run the palletizer over the whole scene and store the program."""
        opts = (JobOptions.fast if fast else JobOptions)(smart_posture=smart_posture)
        res = plan_scene(self.kin(), self.obstacles, self.pallets, opts)
        self.program.steps = res.steps
        return res

    # ---- persistence ------------------------------------------------------
    def scene_dict(self) -> dict:
        return {"obstacles": [b.to_dict() for b in self.obstacles],
                "pallets": [p.to_dict() for p in self.pallets]}

    def write(self) -> str:
        """Write the live project file the app watches. Returns the path."""
        data = project_dict(self.model_name, self._base_pose(),
                            self.program.to_dict(), self.scene_dict())
        tmp = self.live_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self.live_file)                 # atomic — no partial reads
        return self.live_file

    def summary(self) -> str:
        lines = [f"Robot: {self.model_name}",
                 f"Pedestal height: {self._pedestal_height() * _MM:.0f} mm",
                 f"Pallets: {len(self.pallets)}"]
        for p in self.pallets:
            x, y = p.T[0, 3] * _MM, p.T[1, 3] * _MM
            lines.append(f"  · {p.name}: at ({x:.0f}, {y:.0f}) mm, "
                         f"{p.layers} layers, {p.pattern}, {p.total_boxes()} boxes, "
                         f"role={p.role}")
        obs = [b for b in self.obstacles if b.kind != "pedestal"]
        if obs:
            lines.append(f"Obstacles: {', '.join(b.name for b in obs)}")
        lines.append(f"Program steps: {len(self.program.steps)}")
        return "\n".join(lines)


# ===========================================================================
#  MCP wiring (only imported when actually running as a server)
# ===========================================================================
def build_server(session: AppSession):
    # The MCP Python SDK renamed FastMCP → MCPServer in v2. Support both so the
    # server runs whether the installed SDK is v1 or v2 (the tool decorator and
    # .run() are the same on either). Deferred import: optional dependency.
    try:
        from mcp.server.fastmcp import FastMCP as _Server      # mcp 1.x
    except ModuleNotFoundError:
        from mcp.server.mcpserver import MCPServer as _Server   # mcp 2.x

    mcp = _Server("ur-gui")

    def _apply(msg: str) -> str:
        path = session.write()
        return f"{msg}\n(Applied to {path} — the app will reload it.)"

    @mcp.tool()
    def new_scene() -> str:
        """Clear the scene and program to start a fresh simulation."""
        session.clear()
        return _apply("Started a new, empty scene.")

    @mcp.tool()
    def set_robot_model(model: str) -> str:
        """Set the robot arm, e.g. 'UR30' (→ UR30e), 'UR10e', 'UR5e'."""
        name = session.set_model(model)
        return _apply(f"Robot model set to {name}.")

    @mcp.tool()
    def add_pallet(x_mm: float = 600.0, y_mm: float = 0.0, yaw_deg: float = 0.0,
                   length_mm: float = 800.0, width_mm: float = 600.0,
                   box_l_mm: float = 200.0, box_w_mm: float = 150.0,
                   box_h_mm: float = 120.0, box_weight_kg: float = 1.0,
                   layers: int = 3, pattern: str = "column",
                   role: str = "stack", name: str = "") -> str:
        """Add a pallet to stack boxes on. Position (x_mm,y_mm) is the pallet
        centre in the robot base frame (X forward, Y left). pattern is
        'column' | 'interlock' | 'brick'; role is 'stack' | 'source' |
        'destination'."""
        nm = session.add_pallet(x_mm, y_mm, yaw_deg, length_mm, width_mm,
                                box_l_mm=box_l_mm, box_w_mm=box_w_mm,
                                box_h_mm=box_h_mm, box_weight_kg=box_weight_kg,
                                layers=layers, pattern=pattern, role=role,
                                name=name or None)
        return _apply(f"Added pallet '{nm}'.")

    @mcp.tool()
    def add_conveyor(x_mm: float = 400.0, y_mm: float = -400.0,
                     yaw_deg: float = 0.0, length_mm: float = 800.0,
                     width_mm: float = 300.0, height_mm: float = 200.0) -> str:
        """Add a conveyor (a box-spawn / pick surface) the robot picks from."""
        nm = session.add_conveyor(x_mm, y_mm, yaw_deg, length_mm, width_mm, height_mm)
        return _apply(f"Added conveyor '{nm}'.")

    @mcp.tool()
    def add_pedestal(height_mm: float = 300.0, side_mm: float = 400.0) -> str:
        """Mount the robot on a pedestal of the given height (raises the base)."""
        nm = session.add_pedestal(height_mm, side_mm)
        return _apply(f"Added pedestal '{nm}' ({height_mm:.0f} mm tall).")

    @mcp.tool()
    def add_obstacle(x_mm: float, y_mm: float, z_mm: float, length_mm: float,
                     width_mm: float, height_mm: float, name: str = "") -> str:
        """Add a rectangular obstacle the arm must avoid (centre at x,y,z mm)."""
        nm = session.add_obstacle(x_mm, y_mm, z_mm, length_mm, width_mm,
                                  height_mm, name=name or None)
        return _apply(f"Added obstacle '{nm}'.")

    @mcp.tool()
    def plan_palletization(smart_posture: bool = True, fast: bool = False) -> str:
        """Run the palletizer over the whole scene, generate the robot program,
        and report feasibility (reachability + collisions) box by box."""
        res = session.plan(smart_posture=smart_posture, fast=fast)
        path = session.write()
        verdict = "FEASIBLE ✓" if res.ok else "NOT fully feasible ✗"
        return (f"{verdict} — {res.placed_boxes}/{res.total_boxes} boxes, "
                f"{len(res.steps)} program steps.\n{res.message}\n"
                f"(Applied to {path} — the app will reload it.)")

    @mcp.tool()
    def get_scene() -> str:
        """Describe the current scene (robot, pallets, obstacles, program)."""
        return session.summary()

    @mcp.tool()
    def launch_app() -> str:
        """Open the UR GUI Simulator on the live project file (if not already open)."""
        import subprocess
        root = os.path.dirname(os.path.abspath(__file__))
        py = sys.executable
        session.write()
        subprocess.Popen([py, os.path.join(root, "main.py"),
                          "--open", session.live_file], cwd=root)
        return f"Launched the app on {session.live_file}."

    return mcp


# ===========================================================================
#  Self-test (no MCP / Claude needed)
# ===========================================================================
def _selftest() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                   # noqa: BLE001
        pass
    s = AppSession(live_file=os.path.join(
        os.environ.get("TEMP", "."), "mcp_selftest.urgproj"))
    s.set_model("UR30")                                 # → UR30e
    # Two pallets side by side (their near edges ~100 mm apart), like the user's
    # "two pallets 10 cm apart" example. Boxes are presented at the default pick
    # point in front-left of the base.
    s.add_pallet(x_mm=650, y_mm=-350, layers=1, pattern="interlock")
    s.add_pallet(x_mm=650, y_mm=350, layers=1, pattern="interlock")
    print(s.summary())
    print("\nPlanning…")
    res = s.plan(fast=True)
    print(f"ok={res.ok}  {res.placed_boxes}/{res.total_boxes} boxes  "
          f"{len(res.steps)} steps")
    print(res.message[:200])
    path = s.write()
    # verify the written file is a valid, loadable project
    from robot.project import project_from_json
    with open(path, encoding="utf-8") as fh:
        d = project_from_json(fh.read())
    print(f"\nWrote {path}: model={d['model']}, "
          f"pallets={len(d['scene']['pallets'])}, steps={len(d['program']['steps'])}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    _session = AppSession()
    _session.write()                                    # ensure the live file exists
    build_server(_session).run()                        # stdio server for Claude
