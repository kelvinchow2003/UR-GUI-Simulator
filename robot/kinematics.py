"""
robot/kinematics.py
==================================================================
Forward / inverse kinematics and trajectory generation for UR arms.

Everything is implemented in pure NumPy against the standard DH table
in :mod:`robot.ur_models`, so it works with zero extra dependencies.
If ``ikpy`` is installed it is *not* required — the built-in damped
least-squares IK is robust enough for jogging and offline planning.

Conventions
-----------
* Joint vector ``q``      : 6 angles in **radians**.
* Pose (UR convention)   : ``[x, y, z, rx, ry, rz]`` — position in
  **metres**, orientation as an **axis-angle rotation vector** (rad),
  identical to what ``get_actual_tcp_pose`` / ``movel`` use.
* Homogeneous transform  : 4x4 ``numpy.ndarray``.
==================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .ur_models import URModel
from .ur_kinematics_data import UR_URDF, FLANGE_RPY, TOOL0_RPY


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis XYZ Euler (URDF convention) -> 3x3 rotation."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_matrix(x, y, z, roll, pitch, yaw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


def _rotz(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
    return T


# ===========================================================================
#  SO(3) / SE(3) helpers
# ===========================================================================
def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues: axis-angle rotation vector -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=float)
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle rotation vector."""
    R = np.asarray(R, dtype=float)
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-6:
        # near 180deg — use the most stable diagonal element
        idx = int(np.argmax(np.diag(R)))
        axis = np.zeros(3)
        axis[idx] = np.sqrt(max((R[idx, idx] + 1.0) / 2.0, 0.0))
        # fill remaining components
        j, k = (idx + 1) % 3, (idx + 2) % 3
        axis[j] = R[j, idx] / (2 * axis[idx]) if axis[idx] > 1e-9 else 0.0
        axis[k] = R[k, idx] / (2 * axis[idx]) if axis[idx] > 1e-9 else 0.0
        return axis / np.linalg.norm(axis) * angle
    rx = (R[2, 1] - R[1, 2])
    ry = (R[0, 2] - R[2, 0])
    rz = (R[1, 0] - R[0, 1])
    axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
    return axis * angle


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """[x,y,z,rx,ry,rz] -> 4x4 homogeneous transform."""
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> [x,y,z,rx,ry,rz]."""
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:3, 3], matrix_to_rotvec(T[:3, :3])])


def rpy_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convenience: XYZ Euler (rad) -> rotation vector."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return matrix_to_rotvec(Rz @ Ry @ Rx)


# ===========================================================================
#  Kinematic solver
# ===========================================================================
@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    pos_error: float          # metres
    rot_error: float          # radians


class Kinematics:
    """FK/IK/Jacobian for a specific :class:`URModel`."""

    def __init__(self, model: URModel):
        self.model = model
        self.q_min = np.asarray(model.q_min, dtype=float)
        self.q_max = np.asarray(model.q_max, dtype=float)
        self._tcp = np.eye(4)           # tool0 -> TCP offset
        self._base = np.eye(4)          # world -> robot base (mount) transform
        self.base_height = 0.0          # metres the base is lifted (pedestal)
        self._load_urdf(model.name)
        # kept for any legacy reference; kinematics now use the URDF chain
        self.a = np.asarray(model.a, dtype=float)
        self.d = np.asarray(model.d, dtype=float)

    def _load_urdf(self, name: str) -> None:
        """Load UR's exact per-joint origins + tool frames for this model."""
        spec = UR_URDF[name]
        # fixed transform for each joint origin (before the revolute Rz(q_i))
        self._O = [_origin_matrix(*j) for j in spec["joints"]]
        self._flange = _origin_matrix(0, 0, 0, *FLANGE_RPY)   # wrist_3 -> flange
        self._tool0 = _origin_matrix(0, 0, 0, *TOOL0_RPY)     # flange  -> tool0
        self._offsets = spec["offsets"]                       # mesh mount offsets
        self._mesh_paths = spec["paths"]

    # ---- configuration ----------------------------------------------------
    def set_model(self, model: URModel) -> None:
        self.__init__(model)

    def set_tcp(self, pose: Optional[np.ndarray]) -> None:
        """Set the TCP offset from tool0 ([x,y,z,rx,ry,rz])."""
        self._tcp = np.eye(4) if pose is None else pose_to_matrix(pose)

    def set_base_height(self, z: float) -> None:
        """Mount the whole robot ``z`` metres above the floor (e.g. on a
        pedestal). Every FK frame — and therefore the rendered twin, the
        collision capsules, and the world-frame poses IK solves against — is
        lifted by ``z``, so a box placed under the base raises the robot instead
        of being buried in the floor. ``z=0`` restores a floor-mounted robot.

        Only the base's Z is changed; any X/Y position or rotation set via
        :meth:`set_base_pose` (e.g. by dragging the robot's origin gizmo) is
        preserved, so the pedestal height and a repositioned robot coexist."""
        self.base_height = float(z)
        self._base[2, 3] = float(z)

    def set_base_pose(self, T: np.ndarray) -> None:
        """Set the full world→base transform (position + orientation). Used to
        drag/rotate the whole robot around the cell. FK/IK/collision/rendering
        all follow, since every FK frame is premultiplied by this base."""
        self._base = np.asarray(T, float).reshape(4, 4).copy()
        self.base_height = float(self._base[2, 3])

    def base_pose(self) -> np.ndarray:
        """Current world→base transform (position + orientation)."""
        return self._base.copy()

    # ---- forward kinematics (exact UR URDF chain) ------------------------
    def fk_frames(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Link frames for the whole chain::

            [0] base            [4] wrist_1_link
            [1] shoulder_link   [5] wrist_2_link
            [2] upper_arm_link  [6] wrist_3_link
            [3] forearm_link    [7] TCP (tool0 * tcp offset)

        Each joint applies its fixed origin then rotates about local +Z by q_i,
        exactly as UR's URDF does — so these frames drive both the Jacobian and
        the placement of UR's real visual meshes.
        """
        q = np.asarray(q, dtype=float)
        frames = [self._base.copy()]              # base lifted by any mount height
        F = self._base.copy()
        for i in range(6):
            F = F @ self._O[i] @ _rotz(q[i])
            frames.append(F.copy())
        tool0 = F @ self._flange @ self._tool0
        frames.append(tool0 @ self._tcp)          # index 7: TCP
        return frames

    def tool0(self, q: np.ndarray) -> np.ndarray:
        """The UR tool0 frame (flange TCP) without any extra tool offset."""
        q = np.asarray(q, dtype=float)
        F = self._base.copy()
        for i in range(6):
            F = F @ self._O[i] @ _rotz(q[i])
        return F @ self._flange @ self._tool0

    def forward(self, q: np.ndarray, tcp: bool = True) -> np.ndarray:
        """Base->TCP (tcp=True) or base->tool0 (tcp=False) transform."""
        return self.fk_frames(q)[-1] if tcp else self.tool0(q)

    def fk_pose(self, q: np.ndarray, tcp: bool = True) -> np.ndarray:
        """FK returning the UR pose vector [x,y,z,rx,ry,rz]."""
        return matrix_to_pose(self.forward(q, tcp=tcp))

    # ---- mesh mounting helpers -------------------------------------------
    def mesh_offset(self, link: str) -> np.ndarray:
        """4x4 visual mesh offset for a link ('base','shoulder',...,'wrist_3')."""
        return _origin_matrix(*self._offsets[link])

    def mesh_path(self, link: str) -> str:
        return self._mesh_paths[link]

    # ---- Jacobian ---------------------------------------------------------
    def jacobian(self, q: np.ndarray, tcp: bool = True) -> np.ndarray:
        """Geometric Jacobian 6x6 in the base frame (all joints revolute, +Z)."""
        frames = self.fk_frames(q)
        o_end = (frames[-1] if tcp else self.tool0(q))[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            # joint i axis/position = z/origin of the link frame it produces
            z = frames[i + 1][:3, 2]
            o = frames[i + 1][:3, 3]
            J[:3, i] = np.cross(z, o_end - o)
            J[3:, i] = z
        return J

    # ---- inverse kinematics (damped least squares) ------------------------
    def inverse(
        self,
        target_pose: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        tcp: bool = True,
        max_iter: int = 150,
        tol_pos: float = 5e-4,      # 0.5 mm — ample for jogging & planning
        tol_rot: float = 5e-3,      # ~0.3 deg
        damping: float = 0.03,
    ) -> IKResult:
        """
        Solve IK numerically from a seed configuration.

        Returns the joint solution nearest ``q_init`` — ideal for jogging
        and for stitching Cartesian trajectories where continuity matters.
        """
        q = (np.asarray(q_init, dtype=float).copy()
             if q_init is not None else self._mid_config())
        T_target = pose_to_matrix(target_pose)
        p_t = T_target[:3, 3]
        R_t = T_target[:3, :3]

        pos_err = rot_err = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            T = self.forward(q, tcp=tcp)
            p = T[:3, 3]
            R = T[:3, :3]
            e_pos = p_t - p
            e_rot = matrix_to_rotvec(R_t @ R.T)      # orientation error (base frame)
            err = np.concatenate([e_pos, e_rot])
            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))
            if pos_err < tol_pos and rot_err < tol_rot:
                break
            J = self.jacobian(q, tcp=tcp)
            # damped least squares: dq = J^T (J J^T + λ²I)^-1 e
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -0.4, 0.4)              # step limit for stability
            q = q + dq
            q = np.clip(q, self.q_min, self.q_max)

        success = pos_err < tol_pos and rot_err < tol_rot
        return IKResult(q=self._wrap(q), success=success, iterations=it,
                        pos_error=pos_err, rot_error=rot_err)

    def _mid_config(self) -> np.ndarray:
        return np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

    @staticmethod
    def _wrap(q: np.ndarray) -> np.ndarray:
        """Wrap joints into (-pi, pi] for a canonical representation."""
        return (q + np.pi) % (2 * np.pi) - np.pi

    # ---- reachability -----------------------------------------------------
    def in_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.q_min) and np.all(q <= self.q_max))


# ===========================================================================
#  Trajectory generation
# ===========================================================================
class TrajectoryPlanner:
    """Interpolators for joint- and Cartesian-space motions."""

    def __init__(self, kin: Kinematics):
        self.kin = kin

    @staticmethod
    def _quintic(s: np.ndarray) -> np.ndarray:
        """Minimum-jerk time scaling 0->1 (smooth accel at endpoints)."""
        return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5

    def joint_move(self, q0, q1, steps: int = 50) -> np.ndarray:
        """MoveJ interpolation in joint space (quintic time scaling)."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        s = self._quintic(np.linspace(0, 1, steps))
        return q0[None, :] + s[:, None] * (q1 - q0)[None, :]

    def linear_move(self, pose0, pose1, steps: int = 50,
                    seed: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, bool]:
        """
        MoveL: straight-line TCP path with SLERP orientation, solved to
        joints by IK at each sample. Returns (q_path, all_reachable).
        """
        pose0 = np.asarray(pose0, float)
        pose1 = np.asarray(pose1, float)
        p0, p1 = pose0[:3], pose1[:3]
        R0 = rotvec_to_matrix(pose0[3:])
        R1 = rotvec_to_matrix(pose1[3:])
        # relative rotation as axis-angle for SLERP
        dR = matrix_to_rotvec(R1 @ R0.T)

        s = self._quintic(np.linspace(0, 1, steps))
        q_path = np.zeros((steps, 6))
        q_seed = (np.asarray(seed, float) if seed is not None
                  else self.kin.inverse(pose0).q)
        ok = True
        for i, si in enumerate(s):
            p = p0 + si * (p1 - p0)
            R = rotvec_to_matrix(dR * si) @ R0
            pose = np.concatenate([p, matrix_to_rotvec(R)])
            res = self.kin.inverse(pose, q_init=q_seed)
            ok = ok and res.success
            q_path[i] = res.q
            q_seed = res.q
        return q_path, ok

    def time_parameterize(self, q_path: np.ndarray,
                          qd_max: Optional[np.ndarray] = None,
                          dt: float = 0.008) -> np.ndarray:
        """Return timestamps (s) for each waypoint honouring joint speed caps."""
        qd_max = (np.asarray(qd_max) if qd_max is not None
                  else np.asarray(self.kin.model.qd_max))
        times = [0.0]
        for i in range(1, len(q_path)):
            dq = np.abs(q_path[i] - q_path[i - 1])
            seg = float(np.max(dq / qd_max)) if np.any(qd_max > 0) else dt
            times.append(times[-1] + max(seg, dt))
        return np.asarray(times)
