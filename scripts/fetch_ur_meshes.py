#!/usr/bin/env python3
"""
scripts/fetch_ur_meshes.py
==================================================================
Download Universal Robots' **official** visual meshes and convert them
to STL for fast, dependency-light rendering in the 3D viewport.

Source: UniversalRobots/Universal_Robots_ROS2_Description (branch rolling)
        — the same CAD UR ships in their ROS description package.

Meshes are COLLADA (.dae); we tessellate + re-export to .stl once here
(needs ``trimesh`` + ``pycollada``) so the app only ever loads .stl at
runtime. Files land in::

    assets/meshes/<repo>/visual/<link>.stl        (e.g. ur10e/visual/forearm.stl)

Usage:
    python scripts/fetch_ur_meshes.py               # all supported models
    python scripts/fetch_ur_meshes.py UR10e UR5e    # just these
    python scripts/fetch_ur_meshes.py --list        # show models
"""
from __future__ import annotations

import os
import sys
import urllib.request

# allow running from anywhere
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from robot.ur_kinematics_data import UR_URDF          # noqa: E402

RAW_BASE = ("https://raw.githubusercontent.com/UniversalRobots/"
            "Universal_Robots_ROS2_Description/rolling")
# rel paths already begin with "meshes/", so anchor at assets/
ASSETS_DIR = os.path.join(_ROOT, "assets")


def _convert(dae_bytes: bytes, out_stl: str) -> int:
    import trimesh
    import io
    mesh = trimesh.load(io.BytesIO(dae_bytes), file_type="dae", force="mesh")
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:                              # noqa: BLE001
            pass
    mesh.export(out_stl)
    return len(mesh.faces)


def fetch_model(name: str, force: bool = False) -> None:
    spec = UR_URDF[name]
    print(f"\n=== {name} ({spec['repo']}) ===")
    for link, rel in spec["paths"].items():            # rel: meshes/ur10e/visual/base.dae
        out_stl = os.path.join(ASSETS_DIR, rel.replace(".dae", ".stl"))
        os.makedirs(os.path.dirname(out_stl), exist_ok=True)
        if os.path.exists(out_stl) and not force:
            print(f"  {link:10s} skip (exists)")
            continue
        url = f"{RAW_BASE}/{rel}"
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            nfaces = _convert(data, out_stl)
            print(f"  {link:10s} OK  {nfaces:6d} faces  -> {os.path.relpath(out_stl, _ROOT)}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  {link:10s} FAIL  {exc}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv:
        print("Supported models:", ", ".join(UR_URDF))
        return 0
    force = "--force" in argv
    argv = [a for a in argv if not a.startswith("-")]
    models = argv or list(UR_URDF)
    unknown = [m for m in models if m not in UR_URDF]
    if unknown:
        print(f"Unknown model(s): {unknown}\nKnown: {', '.join(UR_URDF)}")
        return 2
    try:
        import trimesh  # noqa: F401
    except Exception:
        print("Requires trimesh + pycollada:  pip install trimesh pycollada")
        return 3
    for m in models:
        fetch_model(m, force=force)
    print("\nDone. Launch the app — real UR meshes load automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
