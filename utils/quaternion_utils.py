"""
Quaternion and rotation utilities (CasADi symbolic + NumPy numeric).

All quaternion conventions follow Hamilton form: q = [qw, qx, qy, qz].
"""

import math
import numpy as np
from casadi import (
    MX, vertcat, horzcat, vertsplit,
    cos, sin, norm_2, if_else, atan2,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Rotation matrices
# ──────────────────────────────────────────────────────────────────────────────

def Rot_zyx(x):
    """ZYX Euler rotation matrix (CasADi symbolic)."""
    phi   = x[3, 0]
    theta = x[4, 0]
    psi   = x[5, 0]

    RotX = MX.zeros(3, 3)
    RotX[0, 0] = 1.0
    RotX[1, 1] = cos(phi)
    RotX[1, 2] = -sin(phi)
    RotX[2, 1] = sin(phi)
    RotX[2, 2] = cos(phi)

    RotY = MX.zeros(3, 3)
    RotY[0, 0] = cos(theta)
    RotY[0, 2] = sin(theta)
    RotY[1, 1] = 1.0
    RotY[2, 0] = -sin(theta)
    RotY[2, 2] = cos(theta)

    RotZ = MX.zeros(3, 3)
    RotZ[0, 0] = cos(psi)
    RotZ[0, 1] = -sin(psi)
    RotZ[1, 0] = sin(psi)
    RotZ[1, 1] = cos(psi)
    RotZ[2, 2] = 1.0

    R = RotZ @ RotY @ RotX
    return R


def QuatToRot(quat):
    """Quaternion → Rotation matrix (CasADi symbolic).

    Parameters
    ----------
    quat : MX (4,1)
        Unit quaternion [qw, qx, qy, qz].

    Returns
    -------
    Rot : MX (3,3)
        Rotation matrix from body to inertial frame.
    """
    q = quat
    q_norm = norm_2(q)
    q_normalized = q / q_norm

    q_hat = MX.zeros(3, 3)
    q_hat[0, 1] = -q_normalized[3]
    q_hat[0, 2] =  q_normalized[2]
    q_hat[1, 2] = -q_normalized[1]
    q_hat[1, 0] =  q_normalized[3]
    q_hat[2, 0] = -q_normalized[2]
    q_hat[2, 1] =  q_normalized[1]

    Rot = MX.eye(3) + 2 * q_hat @ q_hat + 2 * q_normalized[0] * q_hat
    return Rot


# ──────────────────────────────────────────────────────────────────────────────
#  Quaternion algebra (CasADi symbolic)
# ──────────────────────────────────────────────────────────────────────────────

def quaternion_multiply(q1, q2):
    """Hamilton product q1 ⊗ q2 (CasADi symbolic)."""
    w0, x0, y0, z0 = vertsplit(q1)
    w1, x1, y1, z1 = vertsplit(q2)

    scalar_part = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    vector_part = vertcat(
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    )
    return vertcat(scalar_part, vector_part)


def quat_p(quat, omega):
    """Quaternion kinematics: q̇ = 0.5 · q ⊗ [0, ω]."""
    omega_quat = vertcat(MX(0), omega)
    return 0.5 * quaternion_multiply(quat, omega_quat)


def quaternion_error(q_real, quat_d):
    """Quaternion error: q_err = q_real⁻¹ ⊗ q_d."""
    norm_q = norm_2(q_real)
    q_inv = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    return quaternion_multiply(q_inv, quat_d)


def log_cuaternion_casadi(q):
    """Quaternion logarithm on SO(3): Log(q) = 2·θ·q_v/‖q_v‖ (CasADi symbolic).

    Safe at identity: uses atan2(‖q_v‖, q_w)/‖q_v‖ ≈ 1 as q_v→0 via
    a regularised denominator, avoiding 0/0.
    """
    # Enforce positive scalar part (double cover: q and -q represent same rotation)
    q_w = q[0]
    q = if_else(q_w < 0, -q, q)
    q_w = q[0]
    q_v = q[1:]

    norm_q_v = norm_2(q_v)
    theta    = atan2(norm_q_v, q_w)

    # Safe division: when norm_q_v ≈ 0, theta ≈ 0 too, and theta/norm_q_v → 1
    # We regularise with a small epsilon so CasADi generates a finite Jacobian.
    safe_norm = norm_q_v + 1e-9
    log_q = 2 * q_v * theta / safe_norm
    return log_q


# ──────────────────────────────────────────────────────────────────────────────
#  Numeric helpers (NumPy / pure-Python)
# ──────────────────────────────────────────────────────────────────────────────

def euler_to_quaternion(roll, pitch, yaw):
    """Euler (ZYX) → quaternion [qw, qx, qy, qz] (scalar-first)."""
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


def Euler_p(omega, euler):
    """Angular-velocity → Euler-rate transformation."""
    W = np.array([
        [1, np.sin(euler[0]) * np.tan(euler[1]), np.cos(euler[0]) * np.tan(euler[1])],
        [0, np.cos(euler[0]),                    np.sin(euler[0])],
        [0, np.sin(euler[0]) / np.cos(euler[1]), np.cos(euler[0]) / np.cos(euler[1])],
    ])
    return np.dot(W, omega)


def Angulo(ErrAng):
    """Wrap angle to [-π, π]."""
    if ErrAng >= math.pi:
        while ErrAng >= math.pi:
            ErrAng -= 2 * math.pi
    elif ErrAng <= -math.pi:
        while ErrAng <= -math.pi:
            ErrAng += 2 * math.pi
    return ErrAng
