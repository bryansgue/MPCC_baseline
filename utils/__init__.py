"""
utils package – utilities for quadrotor MPCC.

Sub-modules
-----------
casadi_utils : CasADi symbolic functions (rotation matrices, quaternion
               algebra, trajectory interpolation).
numpy_utils  : NumPy / pure-Python functions (Euler conversions, arc-length
               parameterisation, MPCC error decomposition, RK4 integrators).
"""

# ── CasADi symbolic ──────────────────────────────────────────────────────────
from .casadi_utils import (
    rot_zyx_casadi,
    quat_to_rot_casadi,
    quat_multiply_casadi,
    quat_kinematics_casadi,
    quat_error_casadi,
    quat_log_casadi,
    create_position_interpolator_casadi,
    create_tangent_interpolator_casadi,
    create_quat_interpolator_casadi,
)

# ── NumPy / pure-Python ──────────────────────────────────────────────────────
from .numpy_utils import (
    euler_to_quaternion,
    quaternion_to_euler,
    wrap_angle,
    quaternion_hemisphere_correction,
    euler_rate_matrix,
    euler_dot,
    build_arc_length_parameterisation,
    build_waypoints,
    mpcc_errors,
    contouring_lag_scalar,
    rk4_step,
    rk4_step_quadrotor,
    rk4_step_mpcc,
)
