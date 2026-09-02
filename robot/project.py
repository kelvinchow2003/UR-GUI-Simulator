"""
robot/project.py
==================================================================
A **project file** (``.urgproj``) that saves the whole editable state of a
session in one place — so a user can close the app and later re-open exactly
where they left off. Where the per-panel ``.urgui`` file stores only the
Program (the motion steps), a project bundles:

    * the selected robot **model** (e.g. ``UR10e``);
    * the robot **base pose** (pedestal mount height + any repositioning);
    * the **program** (all steps, TCP, default speeds); and
    * the **scene** (every obstacle/conveyor/pedestal + every pallet spec).

It is plain JSON built from each object's own ``to_dict`` / ``from_dict`` (so
the format stays in lock-step with the models and is easy to inspect or diff),
and it is Qt-free — the main window does the gather/apply against the live
objects, this module only does the (de)serialisation.
==================================================================
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                   # avoid import cycles at runtime
    from robot.program import Program
    from robot.scene import SceneModel

FORMAT = "urgui-project"
VERSION = 1
EXTENSION = ".urgproj"


def project_dict(model_name: str, base_pose, program_dict: dict,
                 scene_dict: dict) -> dict:
    """Assemble the project payload from already-serialised parts. Lets non-GUI
    callers (e.g. the MCP server) build a project without a live ``SceneModel``."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "model": str(model_name),
        "base_pose": np.asarray(base_pose, float).reshape(4, 4).tolist(),
        "program": program_dict,
        "scene": scene_dict,
    }


def project_to_json(model_name: str, base_pose, program: "Program",
                    scene: "SceneModel") -> str:
    """Serialise the full session state to an indented JSON string."""
    return json.dumps(
        project_dict(model_name, base_pose, program.to_dict(), scene.to_dict()),
        indent=2)


def project_from_json(text: str) -> dict:
    """Parse and validate a project file, returning its dict.

    Raises :class:`ValueError` with a clear message if the text isn't a UR GUI
    project (so the caller can show it to the user rather than a raw traceback)."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not a valid project file — {exc}") from exc
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise ValueError(
            "This file is not a UR GUI project (.urgproj). "
            "To load just a motion program, use the Program panel's Load (.urgui).")
    if int(d.get("version", 0)) > VERSION:
        raise ValueError(
            f"This project was saved by a newer version (v{d.get('version')}); "
            f"this build understands up to v{VERSION}. Please update the app.")
    return d
