"""
MPCC (Model Predictive Contouring Control) – strict formulation with
symbolic trajectory interpolation.

Key MPCC features:
  • θ (arc-length progress) is an optimisation STATE   (x[13])
  • v_θ (progress velocity) is an optimisation CONTROL (u[4])
  • θ̇ = v_θ  (augmented dynamics)
  • The reference is a **symbolic CasADi function of θ** so the solver
    can differentiate through:
        v_θ  →  θ̇=v_θ  →  θ  →  reference(θ)  →  error  →  cost
    This enables true optimisation of the progress speed v_θ.

Augmented vectors
-----------------
  State   x ∈ ℝ¹⁴ = [p(3), v(3), q(4), ω(3), θ]
  Control u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]

Parameter vector  p ∈ ℝ⁷   (obstacles only — reference comes from θ)
  p[0:4] = [obs_x, obs_y, obs_z, obs_r]
  p[4:7] = [obsmov_x, obsmov_y, obsmov_z]
"""

import numpy as np
import casadi as ca
from casadi import MX, dot, vertcat, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from models.quadrotor_model import (
    MASS, G, JXX, JYY, JZZ,
)
from utils.quaternion_utils import (
    quaternion_error, log_cuaternion_casadi, QuatToRot, quat_p,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CasADi piecewise-linear trajectory interpolation  (θ → reference)
# ══════════════════════════════════════════════════════════════════════════════

def create_casadi_position_interpolator(s_waypoints, pos_waypoints):
    """θ → position [3×1]  (piecewise-linear, differentiable via CasADi)."""
    n = len(s_waypoints)
    s = MX.sym('s')
    s_c = ca.fmin(ca.fmax(s, s_waypoints[0]), s_waypoints[-1])

    px = MX(pos_waypoints[0, -1])
    py = MX(pos_waypoints[1, -1])
    pz = MX(pos_waypoints[2, -1])

    for i in range(n - 2, -1, -1):
        s0, s1 = s_waypoints[i], s_waypoints[i + 1]
        a = (s_c - s0) / (s1 - s0 + 1e-10)
        a = ca.fmin(ca.fmax(a, 0), 1)
        xi = (1 - a) * pos_waypoints[0, i] + a * pos_waypoints[0, i + 1]
        yi = (1 - a) * pos_waypoints[1, i] + a * pos_waypoints[1, i + 1]
        zi = (1 - a) * pos_waypoints[2, i] + a * pos_waypoints[2, i + 1]
        px = ca.if_else(s_c < s1, xi, px)
        py = ca.if_else(s_c < s1, yi, py)
        pz = ca.if_else(s_c < s1, zi, pz)

    return Function('gamma_pos', [s], [vertcat(px, py, pz)])


def create_casadi_tangent_interpolator(s_waypoints, vel_waypoints):
    """θ → unit tangent [3×1]  (piecewise-linear + normalisation)."""
    n = len(s_waypoints)
    s = MX.sym('s')
    s_c = ca.fmin(ca.fmax(s, s_waypoints[0]), s_waypoints[-1])

    tx = MX(vel_waypoints[0, -1])
    ty = MX(vel_waypoints[1, -1])
    tz = MX(vel_waypoints[2, -1])

    for i in range(n - 2, -1, -1):
        s0, s1 = s_waypoints[i], s_waypoints[i + 1]
        a = (s_c - s0) / (s1 - s0 + 1e-10)
        a = ca.fmin(ca.fmax(a, 0), 1)
        txi = (1 - a) * vel_waypoints[0, i] + a * vel_waypoints[0, i + 1]
        tyi = (1 - a) * vel_waypoints[1, i] + a * vel_waypoints[1, i + 1]
        tzi = (1 - a) * vel_waypoints[2, i] + a * vel_waypoints[2, i + 1]
        tx = ca.if_else(s_c < s1, txi, tx)
        ty = ca.if_else(s_c < s1, tyi, ty)
        tz = ca.if_else(s_c < s1, tzi, tz)

    tn = ca.sqrt(tx**2 + ty**2 + tz**2 + 1e-10)
    return Function('gamma_vel', [s], [vertcat(tx / tn, ty / tn, tz / tn)])


def create_casadi_quat_interpolator(s_waypoints, quat_waypoints):
    """θ → quaternion [4×1]  (linear + normalisation, same hemisphere)."""
    n = len(s_waypoints)
    s = MX.sym('s')
    s_c = ca.fmin(ca.fmax(s, s_waypoints[0]), s_waypoints[-1])

    qw = MX(quat_waypoints[0, -1])
    qx = MX(quat_waypoints[1, -1])
    qy = MX(quat_waypoints[2, -1])
    qz = MX(quat_waypoints[3, -1])

    for i in range(n - 2, -1, -1):
        s0, s1 = s_waypoints[i], s_waypoints[i + 1]
        a = (s_c - s0) / (s1 - s0 + 1e-10)
        a = ca.fmin(ca.fmax(a, 0), 1)
        q0, q1_ = quat_waypoints[:, i], quat_waypoints[:, i + 1]
        qwi = (1 - a) * q0[0] + a * q1_[0]
        qxi = (1 - a) * q0[1] + a * q1_[1]
        qyi = (1 - a) * q0[2] + a * q1_[2]
        qzi = (1 - a) * q0[3] + a * q1_[3]
        qw = ca.if_else(s_c < s1, qwi, qw)
        qx = ca.if_else(s_c < s1, qxi, qx)
        qy = ca.if_else(s_c < s1, qyi, qy)
        qz = ca.if_else(s_c < s1, qzi, qz)

    qn = ca.sqrt(qw**2 + qx**2 + qy**2 + qz**2 + 1e-10)
    return Function('gamma_quat', [s], [vertcat(qw / qn, qx / qn, qy / qn, qz / qn)])


# ══════════════════════════════════════════════════════════════════════════════
#  Augmented MPCC model  (θ as state, v_θ as control)
# ══════════════════════════════════════════════════════════════════════════════

def f_system_model_mpcc():
    """Build the augmented quadrotor + θ model.

    Returns  model, f_system, f_x, g_x
    """
    model_name = 'Drone_ode_complete'
    m = MASS;  e3 = MX([0, 0, 1]);  g = G

    # States (14)
    p1 = MX.sym('p1');  p2 = MX.sym('p2');  p3 = MX.sym('p3')
    v1 = MX.sym('v1');  v2 = MX.sym('v2');  v3 = MX.sym('v3')
    q0 = MX.sym('q0');  q1 = MX.sym('q1');  q2 = MX.sym('q2');  q3 = MX.sym('q3')
    w1 = MX.sym('w1');  w2 = MX.sym('w2');  w3 = MX.sym('w3')
    theta = MX.sym('theta')
    x = vertcat(p1, p2, p3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3, theta)

    # Controls (5)
    Tt = MX.sym('Tt');  tau1 = MX.sym('tau1');  tau2 = MX.sym('tau2');  tau3 = MX.sym('tau3')
    v_theta = MX.sym('v_theta')
    u = vertcat(Tt, tau1, tau2, tau3, v_theta)

    # xdot
    p1_p = MX.sym('p1_p');  p2_p = MX.sym('p2_p');  p3_p = MX.sym('p3_p')
    v1_p = MX.sym('v1_p');  v2_p = MX.sym('v2_p');  v3_p = MX.sym('v3_p')
    q0_p = MX.sym('q0_p');  q1_p = MX.sym('q1_p');  q2_p = MX.sym('q2_p');  q3_p = MX.sym('q3_p')
    w1_p = MX.sym('w1_p');  w2_p = MX.sym('w2_p');  w3_p = MX.sym('w3_p')
    theta_p = MX.sym('theta_p')
    x_p = vertcat(p1_p, p2_p, p3_p, v1_p, v2_p, v3_p,
                  q0_p, q1_p, q2_p, q3_p, w1_p, w2_p, w3_p, theta_p)

    # Dynamics
    from casadi import inv, cross, horzcat
    quat = vertcat(q0, q1, q2, q3);  w = vertcat(w1, w2, w3);  Rot = QuatToRot(quat)
    I_inertia = vertcat(
        horzcat(MX(JXX), MX(0), MX(0)),
        horzcat(MX(0), MX(JYY), MX(0)),
        horzcat(MX(0), MX(0), MX(JZZ)),
    )
    dp = vertcat(v1, v2, v3)
    dv = -e3 * g + (Rot @ vertcat(MX(0), MX(0), Tt)) / m
    dq = quat_p(quat, w)
    dw = inv(I_inertia) @ (vertcat(tau1, tau2, tau3) - cross(w, I_inertia @ w))
    dtheta = v_theta

    f_expl = vertcat(dp, dv, dq, dw, dtheta)

    # Auxiliary functions
    from casadi import substitute, jacobian
    u_zero = MX.zeros(u.size1(), 1)
    f_x_func = Function('f0', [x], [substitute(f_expl, u, u_zero)])
    g_x_func = Function('g',  [x], [jacobian(f_expl, u)])
    f_system = Function('system', [x, u], [f_expl])

    # Parameters: obstacles only (7)
    obs_x = MX.sym('obs_x');  obs_y = MX.sym('obs_y');  obs_z = MX.sym('obs_z');  obs_r = MX.sym('obs_r')
    obsmov_x = MX.sym('obsmov_x');  obsmov_y = MX.sym('obsmov_y');  obsmov_z = MX.sym('obsmov_z')
    p_param = vertcat(obs_x, obs_y, obs_z, obs_r, obsmov_x, obsmov_y, obsmov_z)

    # Model
    model = AcadosModel()
    model.f_impl_expr = x_p - f_expl
    model.f_expl_expr = f_expl
    model.x = x;  model.xdot = x_p;  model.u = u;  model.p = p_param
    model.name = model_name

    return model, f_system, f_x_func, g_x_func


# ──────────────────────────────────────────────────────────────────────────────
#  Default cost weights (from dual-quaternion MPCC example)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_Q_Q    = [1, 1, 1]                       # Quaternion log error
DEFAULT_Q_EL   = 5.0                             # Lag error (scalar weight)
DEFAULT_Q_EC   = [10, 10, 10]                    # Contouring error
DEFAULT_U_MAT  = [0.1, 250, 250, 250]            # Control effort (deviation from hover)
DEFAULT_Q_OMEGA = 0.5                            # Angular velocity
DEFAULT_Q_S    = 0.3                             # Progress: Q_s*(v_max-v_θ)²

DEFAULT_T_MAX      = 2 * 8.5
DEFAULT_T_MIN      = 0.0
DEFAULT_TAUX_MAX   = 0.025
DEFAULT_TAUY_MAX   = 0.021
DEFAULT_TAUZ_MAX   = 0.02
DEFAULT_VTHETA_MIN = 0.0
DEFAULT_VTHETA_MAX = 15.0


# ══════════════════════════════════════════════════════════════════════════════
#  OCP builder
# ══════════════════════════════════════════════════════════════════════════════

def create_mpcc_ocp_description(
    x0, N_horizon, t_horizon, s_max,
    gamma_pos, gamma_vel, gamma_quat,
) -> AcadosOcp:
    """Create strict MPCC OCP with symbolic trajectory interpolation.

    The reference sd(θ), tangent(θ), qd(θ) are CasADi functions of the
    state θ=x[13].  The solver differentiates through θ→reference→error,
    which makes v_θ truly optimisable.
    """
    ocp = AcadosOcp()
    model, _, _, _ = f_system_model_mpcc()
    ocp.model = model

    np_param = model.p.size()[0]   # 7

    ocp.solver_options.N_horizon = N_horizon
    ocp.parameter_values = np.zeros(np_param)

    # Weights
    Q_q     = np.diag(DEFAULT_Q_Q)
    Q_el    = DEFAULT_Q_EL                    # scalar: penalises (t·e_t)²
    Q_ec    = np.diag(DEFAULT_Q_EC)
    U_mat   = np.diag(DEFAULT_U_MAT)
    Q_omega = DEFAULT_Q_OMEGA
    Q_s     = DEFAULT_Q_S

    T_hover = MASS * 9.81                     # hover thrust [N] ≈ 9.81

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ── Symbolic reference from θ ────────────────────────────────────────
    theta_state = model.x[13]
    sd      = gamma_pos(theta_state)      # [3]  reference position
    tangent = gamma_vel(theta_state)      # [3]  unit tangent
    qd      = gamma_quat(theta_state)     # [4]  desired quaternion

    # ── Errors ───────────────────────────────────────────────────────────
    quat_err = quaternion_error(model.x[6:10], qd)
    log_q    = log_cuaternion_casadi(quat_err)

    e_t = sd - model.x[0:3]            # 3-vector: reference minus position

    # Lag error (scalar projection onto tangent, then squared)
    e_lag    = dot(tangent, e_t)        # scalar  ∈ ℝ
    # Contouring error (component orthogonal to tangent)
    P_ec = MX.eye(3) - tangent @ tangent.T
    ec   = P_ec @ e_t                  # 3-vector ⊥ tangent

    omega   = model.x[10:13]
    v_theta = model.u[4]

    # ── Nominal (hover) control vector — penalise deviation, not absolute ─
    u_nominal = vertcat(MX(T_hover), MX(0), MX(0), MX(0))
    u_dev     = model.u[0:4] - u_nominal

    # ── Cost terms ───────────────────────────────────────────────────────
    control_cost      = u_dev.T @ U_mat @ u_dev
    actitud_cost      = log_q.T @ Q_q @ log_q
    error_contorno    = ec.T @ Q_ec @ ec
    error_lag         = Q_el * e_lag**2             # scalar weight × scalar²
    omega_cost        = Q_omega * (omega.T @ omega)
    arc_speed_penalty = Q_s * (DEFAULT_VTHETA_MAX - v_theta)**2

    # ── Stage cost ───────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost = (
        error_contorno + error_lag + actitud_cost
        + control_cost + omega_cost
        + arc_speed_penalty
    )

    # ── Terminal cost ────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost_e = (
        error_contorno + error_lag + actitud_cost + omega_cost
    )

    # ── Constraints ──────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([
        DEFAULT_T_MIN, -DEFAULT_TAUX_MAX, -DEFAULT_TAUY_MAX, -DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MIN,
    ])
    ocp.constraints.ubu = np.array([
        DEFAULT_T_MAX, DEFAULT_TAUX_MAX, DEFAULT_TAUY_MAX, DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MAX,
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.lbx   = np.array([0.0])
    ocp.constraints.ubx   = np.array([s_max])
    ocp.constraints.idxbx = np.array([13])

    ocp.constraints.x0 = x0

    # ── Solver options ───────────────────────────────────────────────────
    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI"
    ocp.solver_options.tol              = 1e-4
    ocp.solver_options.tf               = t_horizon

    return ocp


# ══════════════════════════════════════════════════════════════════════════════
#  Solver factory
# ══════════════════════════════════════════════════════════════════════════════

def build_mpcc_solver(x0, N_prediction, t_prediction, s_max,
                      gamma_pos, gamma_vel, gamma_quat,
                      use_cython=True):
    """Create, code-generate, compile and return the MPCC solver."""
    ocp = create_mpcc_ocp_description(
        x0, N_prediction, t_prediction, s_max,
        gamma_pos, gamma_vel, gamma_quat,
    )
    model = ocp.model
    _, f_system, _, _ = f_system_model_mpcc()

    solver_json = 'acados_ocp_' + model.name + '.json'

    if use_cython:
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, ocp, model, f_system
