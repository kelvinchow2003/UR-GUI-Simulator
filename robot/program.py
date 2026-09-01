"""
robot/program.py
==================================================================
The visual program model and the two code generators that turn it
into runnable code.

A :class:`Program` is an ordered list of :class:`ProgramStep` objects.
The Program Builder panel edits this model; the code editor renders it
through :class:`URScriptGenerator` and :class:`PythonGenerator`; and the
offline simulator / executor walks the same steps. One model, many views.

Every step is JSON-serialisable so programs can be saved/loaded (.urgui).
==================================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class StepType(str, Enum):
    MOVEJ = "MoveJ"            # joint-space move to a joint target
    MOVEL = "MoveL"           # linear TCP move to a pose
    MOVEP = "MoveP"           # process (constant-speed blended) move
    PROCESS = "ProcessPoint"  # a via point on a CAD-derived toolpath
    GRIPPER_OPEN = "GripperOpen"
    GRIPPER_CLOSE = "GripperClose"
    DELAY = "Delay"
    SET_DO = "SetDigitalOut"
    WAIT_DI = "WaitDigitalIn"
    COMMENT = "Comment"


# joint-target step types vs pose-target step types
_JOINT_STEPS = {StepType.MOVEJ}
_POSE_STEPS = {StepType.MOVEL, StepType.MOVEP, StepType.PROCESS}


@dataclass
class ProgramStep:
    """One line in the program tree."""
    type: StepType
    name: str = ""
    # motion targets — only the relevant one is populated per type
    q: Optional[List[float]] = None            # joint target (rad), len 6
    pose: Optional[List[float]] = None         # TCP pose [x,y,z,rx,ry,rz]
    # motion params
    speed: float = 1.0                         # rad/s (joint) or m/s (cart)
    accel: float = 1.2                         # rad/s^2 or m/s^2
    blend: float = 0.0                         # blend radius (m)
    # I/O + timing params
    duration: float = 1.0                      # for DELAY (s)
    pin: int = 0                               # for SET_DO / WAIT_DI
    value: bool = True                         # for SET_DO
    text: str = ""                             # for COMMENT
    enabled: bool = True
    uid: int = field(default_factory=lambda: ProgramStep._next_uid())

    _uid_counter = 0

    @classmethod
    def _next_uid(cls) -> int:
        cls._uid_counter += 1
        return cls._uid_counter

    # ---- presentation -----------------------------------------------------
    def label(self) -> str:
        if self.name:
            return self.name
        t = self.type
        if t in _JOINT_STEPS and self.q is not None:
            deg = ", ".join(f"{np.degrees(v):.1f}" for v in self.q)
            return f"{t.value}  [{deg}]°"
        if t in _POSE_STEPS and self.pose is not None:
            mm = ", ".join(f"{v*1000:.0f}" for v in self.pose[:3])
            return f"{t.value}  ({mm}) mm"
        if t is StepType.DELAY:
            return f"Delay {self.duration:g}s"
        if t is StepType.SET_DO:
            return f"DO[{self.pin}] = {'ON' if self.value else 'OFF'}"
        if t is StepType.WAIT_DI:
            return f"Wait DI[{self.pin}]"
        if t is StepType.COMMENT:
            return f"# {self.text}"
        return t.value

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        d.pop("_uid_counter", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProgramStep":
        d = dict(d)
        d["type"] = StepType(d["type"])
        d.pop("uid", None)
        return cls(**d)


@dataclass
class Program:
    """Ordered container of steps plus metadata (TCP, default speeds)."""
    name: str = "Untitled"
    steps: List[ProgramStep] = field(default_factory=list)
    tcp: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    default_j_speed: float = 1.05
    default_j_accel: float = 1.40
    default_l_speed: float = 0.25
    default_l_accel: float = 1.20

    # ---- list ops used by the tree view -----------------------------------
    def add(self, step: ProgramStep, index: Optional[int] = None) -> None:
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)

    def remove(self, uid: int) -> None:
        self.steps = [s for s in self.steps if s.uid != uid]

    def move(self, uid: int, delta: int) -> None:
        idx = self._index(uid)
        if idx is None:
            return
        new = max(0, min(len(self.steps) - 1, idx + delta))
        self.steps.insert(new, self.steps.pop(idx))

    def _index(self, uid: int) -> Optional[int]:
        for i, s in enumerate(self.steps):
            if s.uid == uid:
                return i
        return None

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tcp": list(self.tcp),
            "default_j_speed": self.default_j_speed,
            "default_j_accel": self.default_j_accel,
            "default_l_speed": self.default_l_speed,
            "default_l_accel": self.default_l_accel,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Program":
        prog = cls(
            name=d.get("name", "Untitled"),
            tcp=d.get("tcp", [0, 0, 0, 0, 0, 0]),
            default_j_speed=d.get("default_j_speed", 1.05),
            default_j_accel=d.get("default_j_accel", 1.40),
            default_l_speed=d.get("default_l_speed", 0.25),
            default_l_accel=d.get("default_l_accel", 1.20),
        )
        prog.steps = [ProgramStep.from_dict(s) for s in d.get("steps", [])]
        return prog

    def load_from(self, other: "Program") -> None:
        """Copy another program's contents *in place*, so panels holding a
        reference to this instance see the update without being rebuilt."""
        self.name = other.name
        self.tcp = list(other.tcp)
        self.steps = other.steps
        self.default_j_speed = other.default_j_speed
        self.default_j_accel = other.default_j_accel
        self.default_l_speed = other.default_l_speed
        self.default_l_accel = other.default_l_accel

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Program":
        return cls.from_dict(json.loads(text))


# ===========================================================================
#  Code generators
# ===========================================================================
def _fmt_list(vals, prec=6) -> str:
    return "[" + ", ".join(f"{v:.{prec}f}" for v in vals) + "]"


class URScriptGenerator:
    """Render a :class:`Program` as a native ``.script`` program."""

    def generate(self, prog: Program) -> str:
        L: List[str] = []
        L.append(f"# URScript generated from '{prog.name}' by UR GUI Simulator")
        L.append("def ur_gui_program():")
        ind = "    "
        if any(prog.tcp):
            L.append(f"{ind}set_tcp(p{_fmt_list(prog.tcp)})")
        for s in prog.steps:
            if not s.enabled:
                L.append(f"{ind}# (disabled) {s.label()}")
                continue
            L.append(f"{ind}{self._line(s, prog)}")
        L.append("end")
        L.append("")
        L.append("ur_gui_program()")
        return "\n".join(L)

    def _line(self, s: ProgramStep, prog: Program) -> str:
        t = s.type
        if t is StepType.MOVEJ:
            return (f"movej({_fmt_list(s.q)}, a={s.accel:g}, v={s.speed:g}"
                    f"{self._blend(s)})")
        if t is StepType.MOVEL:
            return (f"movel(p{_fmt_list(s.pose)}, a={s.accel:g}, v={s.speed:g}"
                    f"{self._blend(s)})")
        if t in (StepType.MOVEP, StepType.PROCESS):
            r = s.blend if s.blend > 0 else 0.005
            return f"movep(p{_fmt_list(s.pose)}, a={s.accel:g}, v={s.speed:g}, r={r:g})"
        if t is StepType.GRIPPER_OPEN:
            return "set_digital_out(0, False)  # gripper open"
        if t is StepType.GRIPPER_CLOSE:
            return "set_digital_out(0, True)  # gripper close"
        if t is StepType.DELAY:
            return f"sleep({s.duration:g})"
        if t is StepType.SET_DO:
            return f"set_standard_digital_out({s.pin}, {str(s.value)})"
        if t is StepType.WAIT_DI:
            return (f"while (get_standard_digital_in({s.pin}) == False):\n"
                    f"        sync()\n    end")
        if t is StepType.COMMENT:
            return f"# {s.text}"
        return f"# unsupported: {t.value}"

    @staticmethod
    def _blend(s: ProgramStep) -> str:
        return f", r={s.blend:g}" if s.blend > 0 else ""


class PythonGenerator:
    """Render a :class:`Program` as a runnable ``ur_rtde`` Python script."""

    def generate(self, prog: Program, ip: str = "192.168.1.100") -> str:
        # A "Wait DI" step reads inputs, which needs the receive interface.
        needs_receive = any(
            s.enabled and s.type is StepType.WAIT_DI for s in prog.steps)
        L: List[str] = []
        L.append('"""Auto-generated by UR GUI Simulator — ur_rtde control script."""')
        L.append("import time")
        L.append("from rtde_control import RTDEControlInterface as RTDEControl")
        L.append("from rtde_io import RTDEIOInterface as RTDEIO")
        if needs_receive:
            L.append("from rtde_receive import RTDEReceiveInterface as RTDEReceive")
        L.append("")
        L.append(f'ROBOT_IP = "{ip}"')
        L.append("rtde_c = RTDEControl(ROBOT_IP)")
        L.append("rtde_io = RTDEIO(ROBOT_IP)")
        if needs_receive:
            L.append("rtde_r = RTDEReceive(ROBOT_IP)")
        if any(prog.tcp):
            L.append(f"rtde_c.setTcp({_fmt_list(prog.tcp)})")
        L.append("")
        L.append("try:")
        ind = "    "
        for s in prog.steps:
            if not s.enabled:
                L.append(f"{ind}# (disabled) {s.label()}")
                continue
            for line in self._lines(s):
                L.append(f"{ind}{line}")
        L.append("finally:")
        L.append(f"{ind}rtde_c.stopScript()")
        L.append("")
        return "\n".join(L)

    def _lines(self, s: ProgramStep) -> List[str]:
        t = s.type
        if t is StepType.MOVEJ:
            return [f"rtde_c.moveJ({_fmt_list(s.q)}, {s.speed:g}, {s.accel:g})"]
        if t is StepType.MOVEL:
            return [f"rtde_c.moveL({_fmt_list(s.pose)}, {s.speed:g}, {s.accel:g})"]
        if t in (StepType.MOVEP, StepType.PROCESS):
            r = s.blend if s.blend > 0 else 0.005
            return [f"rtde_c.moveL({_fmt_list(s.pose)}, {s.speed:g}, {s.accel:g})  # process r={r:g}"]
        if t is StepType.GRIPPER_OPEN:
            return ["rtde_io.setStandardDigitalOut(0, False)  # gripper open"]
        if t is StepType.GRIPPER_CLOSE:
            return ["rtde_io.setStandardDigitalOut(0, True)  # gripper close"]
        if t is StepType.DELAY:
            return [f"time.sleep({s.duration:g})"]
        if t is StepType.SET_DO:
            return [f"rtde_io.setStandardDigitalOut({s.pin}, {s.value})"]
        if t is StepType.WAIT_DI:
            return [f"while not rtde_r.getDigitalInState({s.pin}):",
                    "    time.sleep(0.01)"]
        if t is StepType.COMMENT:
            return [f"# {s.text}"]
        return [f"# unsupported: {t.value}"]
