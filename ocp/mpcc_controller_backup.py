"""
MPCC (Model Predictive Contouring Control) – strict formulation.

Key difference from a standard tracker:
  • The arc-length progress θ is an **optimisation state**   (x[13])
  • The progress velocity  v_θ is an **optimisation control** (u[4])
  • The dynamics are augmented with  θ̇ = v_θ
  • The solver *maximises* v_θ (equivalently minimises −v_θ) to push the UAV
    forward along the path as fast as possible while keeping contouring and
    lag errors small.

Augmented vectors
-----------------
  State   x ∈ ℝ¹⁴ = [p(3), v(3), q(4), ω(3), θ]
  Control u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]

Parameter vector  p ∈ ℝ²⁴
  p[0:3]   = sd      (reference position on path, evaluated at predicted θ)
  p[3:6]   = sd_p    (unit tangent at that point)
  p[6:10]  = qd      (desired quaternion – from tangent heading)
  p[10:13] = wd      (desired angular velocity, typically 0)
  p[13:17] = ud      (desired control, for regularisation)
  p[17:21] = [obs_x, obs_y, obs_z, obs_r]   (closest static obstacle)
  p[21:24] = [obsmov_x, obsmov_y, obsmov_z] (mobile obstacle centre)
"""

import numpy as np
from casadi import MX, SX, dot, vertcat, Function
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from models.quadrotor_model import (
    f_system_model,
    MASS, G, JXX, JYY, JZZ,
)
from utils.casadi_utils import (
    quat_error_casadi as quaternion_error,
    quat_log_casadi as log_cuaternion_casadi,
    quat_to_rot_casadi as QuatToRot,
    quat_kinematics_casadi as quat_p,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Augmented MPCC model  (θ as state, v_θ as control)
# ──────────────────────────────────────────────────────────────────────────────

def f_system_model_mpcc():
    """Build the augmented quadrotor model for strict MPCC.

    State   x ∈ ℝ¹⁴ = [p(3), v(3), q(4), ω(3), θ]
    Control u ∈ ℝ⁵  = [T, τx, τy, τz, v_θ]
    Param   p ∈ ℝ²⁴ (17 ref/path  + 7 obstacle)

    Returns
    -------
    model      : AcadosModel   – augmented model
    f_system   : casadi.Function(x14, u5) → ẋ14
    f_x        : casadi.Function(x14) → f₀(x14)     (drift, u=0)
    g_x        : casadi.Function(x14) → ∂f/∂u        (input matrix)
    """

    model_name = 'Drone_ode_complete'

    m = MASS
    e3 = MX([0, 0, 1])
    g  = G

    # ── States (14) ──────────────────────────────────────────────────────
    p1 = MX.sym('p1');  p2 = MX.sym('p2');  p3 = MX.sym('p3')
    v1 = MX.sym('v1');  v2 = MX.sym('v2');  v3 = MX.sym('v3')
    q0 = MX.sym('q0');  q1 = MX.sym('q1');  q2 = MX.sym('q2');  q3 = MX.sym('q3')
    w1 = MX.sym('w1');  w2 = MX.sym('w2');  w3 = MX.sym('w3')
    theta = MX.sym('theta')   # arc-length progress

    x = vertcat(p1, p2, p3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3, theta)

    # ── Controls (5) ─────────────────────────────────────────────────────
    Tt   = MX.sym('Tt')
    tau1 = MX.sym('tau1');  tau2 = MX.sym('tau2');  tau3 = MX.sym('tau3')
    v_theta = MX.sym('v_theta')   # progress velocity

    u = vertcat(Tt, tau1, tau2, tau3, v_theta)

    # ── xdot (implicit form) ────────────────────────────────────────────
    p1_p = MX.sym('p1_p');  p2_p = MX.sym('p2_p');  p3_p = MX.sym('p3_p')
    v1_p = MX.sym('v1_p');  v2_p = MX.sym('v2_p');  v3_p = MX.sym('v3_p')
    q0_p = MX.sym('q0_p');  q1_p = MX.sym('q1_p');  q2_p = MX.sym('q2_p');  q3_p = MX.sym('q3_p')
    w1_p = MX.sym('w1_p');  w2_p = MX.sym('w2_p');  w3_p = MX.sym('w3_p')
    theta_p = MX.sym('theta_p')

    x_p = vertcat(p1_p, p2_p, p3_p, v1_p, v2_p, v3_p,
                  q0_p, q1_p, q2_p, q3_p, w1_p, w2_p, w3_p, theta_p)

    # ── Quadrotor dynamics (same as base model) ─────────────────────────
    from casadi import inv, cross, horzcat

    quat = vertcat(q0, q1, q2, q3)
    w    = vertcat(w1, w2, w3)
    Rot  = QuatToRot(quat)

    I_inertia = vertcat(
        horzcat(MX(JXX), MX(0), MX(0)),
        horzcat(MX(0), MX(JYY), MX(0)),
        horzcat(MX(0), MX(0), MX(JZZ)),
    )

    u_thrust = vertcat(MX(0), MX(0), Tt)
    u_torque = vertcat(tau1, tau2, tau3)

    dp = vertcat(v1, v2, v3)
    dv = -e3 * g + (Rot @ u_thrust) / m
    dq = quat_p(quat, w)
    dw = inv(I_inertia) @ (u_torque - cross(w, I_inertia @ w))
    dtheta = v_theta                        # ← NEW: θ̇ = v_θ

    f_expl = vertcat(dp, dv, dq, dw, dtheta)

    # ── Auxiliary CasADi functions ───────────────────────────────────────
    from casadi import substitute, jacobian

    u_zero   = MX.zeros(u.size1(), 1)
    f0_expr  = substitute(f_expl, u, u_zero)

    f_x_func = Function('f0', [x], [f0_expr])
    g_x_func = Function('g',  [x], [jacobian(f_expl, u)])
    f_system = Function('system', [x, u], [f_expl])

    # ── Parameter vector (17 ref + 7 obstacle = 24) ─────────────────────
    p1_d = MX.sym('p1_d');  p2_d = MX.sym('p2_d');  p3_d = MX.sym('p3_d')
    v1_d = MX.sym('v1_d');  v2_d = MX.sym('v2_d');  v3_d = MX.sym('v3_d')
    q0_d = MX.sym('q0_d');  q1_d = MX.sym('q1_d');  q2_d = MX.sym('q2_d');  q3_d = MX.sym('q3_d')
    w1_d = MX.sym('w1_d');  w2_d = MX.sym('w2_d');  w3_d = MX.sym('w3_d')
    T_d  = MX.sym('T_d');   tau1_d = MX.sym('tau1_d')
    tau2_d = MX.sym('tau2_d');  tau3_d = MX.sym('tau3_d')

    p_ref = vertcat(p1_d, p2_d, p3_d, v1_d, v2_d, v3_d,
                    q0_d, q1_d, q2_d, q3_d, w1_d, w2_d, w3_d,
                    T_d, tau1_d, tau2_d, tau3_d)            # 17

    obs_x   = MX.sym('obs_x');   obs_y   = MX.sym('obs_y')
    obs_z   = MX.sym('obs_z');   obs_r   = MX.sym('obs_r')
    obsmov_x = MX.sym('obsmov_x');  obsmov_y = MX.sym('obsmov_y')
    obsmov_z = MX.sym('obsmov_z')

    p_obs = vertcat(obs_x, obs_y, obs_z, obs_r,
                    obsmov_x, obsmov_y, obsmov_z)           # 7

    p_param = vertcat(p_ref, p_obs)                         # 24

    # ── Acados model ─────────────────────────────────────────────────────
    f_impl = x_p - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x    = x
    model.xdot = x_p
    model.u    = u
    model.p    = p_param
    model.name = model_name

    return model, f_system, f_x_func, g_x_func


# ──────────────────────────────────────────────────────────────────────────────
#  Default cost weights
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_Q_Q    = [1, 1, 1]                       # Quaternion log error
DEFAULT_Q_EL   = [50, 50, 50]                    # Lag error  (along-path, couples θ with position)
DEFAULT_Q_EC   = [100, 100, 100]                 # Contouring error (cross-path)
DEFAULT_U_MAT  = [0.1, 250, 250, 250]            # Control effort (4 physical inputs only)
DEFAULT_Q_OMEGA = 0.5                            # Angular velocity penalty
DEFAULT_Q_S    = 0.3                             # Progress speed weight  Q_s*(v_max - v_θ)²

# ──────────────────────────────────────────────────────────────────────────────
#  Default actuation limits
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_T_MAX      = 2 * 8.5   # [N]
DEFAULT_T_MIN      = 0.0       # [N]
DEFAULT_TAUX_MAX   = 0.025     # [N·m]
DEFAULT_TAUY_MAX   = 0.021     # [N·m]
DEFAULT_TAUZ_MAX   = 0.02      # [N·m]
DEFAULT_VTHETA_MIN = 0.0       # [m/s along arc]  — don't go backwards
DEFAULT_VTHETA_MAX = 15.0      # [m/s along arc]  — maximum progress speed


# ──────────────────────────────────────────────────────────────────────────────
#  OCP builder
# ──────────────────────────────────────────────────────────────────────────────

def create_mpcc_ocp_description(
    x0,
    N_horizon,
    t_horizon,
    s_max=None,
) -> AcadosOcp:
    """Create the strict MPCC acados OCP description.

    Parameters
    ----------
    x0        : ndarray (14,)  – initial state [p,v,q,ω,θ₀]
    N_horizon : int            – prediction steps
    t_horizon : float          – prediction time [s]
    s_max     : float or None  – total arc length (for θ upper bound)

    Returns
    -------
    ocp : AcadosOcp
    """
    ocp = AcadosOcp()

    model, f_system, f_x, g_x = f_system_model_mpcc()
    ocp.model = model

    nx = model.x.size()[0]    # 14
    nu = model.u.size()[0]    # 5
    np_param = model.p.size()[0]  # 24

    # ── Horizon ──────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N_horizon

    # ── Parameter values ─────────────────────────────────────────────────
    ocp.parameter_values = np.zeros(np_param)

    # ── Cost matrices ────────────────────────────────────────────────────
    Q_q      = np.diag(DEFAULT_Q_Q)
    Q_el     = np.diag(DEFAULT_Q_EL)
    Q_ec     = np.diag(DEFAULT_Q_EC)
    U_mat    = np.diag(DEFAULT_U_MAT)        # 4×4 (physical inputs only)
    Q_omega  = DEFAULT_Q_OMEGA
    Q_s      = DEFAULT_Q_S

    # ── External cost ────────────────────────────────────────────────────
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # Quaternion error
    quat_err = quaternion_error(model.x[6:10], model.p[6:10])
    log_q    = log_cuaternion_casadi(quat_err)

    # Reference position on path (from parameter, set per-node)
    sd   = model.p[0:3]
    e_t  = sd - model.x[0:3]

    # Unit tangent (from parameter)
    sd_p = model.p[3:6]
    tangent = sd_p

    # Lag error  (projection of position error onto tangent)
    el = dot(tangent, e_t) * tangent

    # Contouring error  (perpendicular component)
    I3   = MX.eye(3)
    P_ec = I3 - tangent @ tangent.T
    ec   = P_ec @ e_t

    # v_θ — progress velocity (last control input)
    v_theta = model.u[4]

    # Angular velocity (for smoothness penalty)
    omega = model.x[10:13]

    # Cost terms
    control_cost       = model.u[0:4].T @ U_mat @ model.u[0:4]     # 4 physical inputs
    actitud_cost       = log_q.T @ Q_q @ log_q
    error_contorno     = ec.T @ Q_ec @ ec
    error_lag          = el.T @ Q_el @ el
    omega_cost         = Q_omega * (omega.T @ omega)

    # Progress incentive (QUADRATIC — pushes v_θ toward its upper bound)
    # This is the key MPCC term: Q_s * (v_θ_max - v_θ)²
    # - Proper curvature for Gauss-Newton Hessian approximation
    # - When v_θ = v_θ_max → penalty = 0 (maximum speed, ideal)
    # - When v_θ < v_θ_max → penalty > 0 (slower, penalised)
    # - But if v_θ is too high → lag error grows → cost increases
    # - The solver finds the OPTIMAL v_θ balancing speed vs tracking accuracy
    arc_speed_penalty  = Q_s * (DEFAULT_VTHETA_MAX - v_theta)**2

    # ── Stage cost ───────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost = (
        error_contorno + error_lag + actitud_cost
        + control_cost + omega_cost
        + arc_speed_penalty
    )

    # ── Terminal cost ────────────────────────────────────────────────────
    ocp.model.cost_expr_ext_cost_e = (
        error_contorno + error_lag + actitud_cost
        + omega_cost
    )

    # ── Control constraints (box) — 5 inputs ────────────────────────────
    ocp.constraints.lbu = np.array([
        DEFAULT_T_MIN, -DEFAULT_TAUX_MAX, -DEFAULT_TAUY_MAX, -DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MIN,
    ])
    ocp.constraints.ubu = np.array([
        DEFAULT_T_MAX, DEFAULT_TAUX_MAX, DEFAULT_TAUY_MAX, DEFAULT_TAUZ_MAX,
        DEFAULT_VTHETA_MAX,
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # ── State constraint on θ (don't exceed total arc length) ────────────
    if s_max is not None:
        ocp.constraints.lbx = np.array([0.0])
        ocp.constraints.ubx = np.array([s_max])
        ocp.constraints.idxbx = np.array([13])    # θ index

    # ── Initial state constraint ─────────────────────────────────────────
    ocp.constraints.x0 = x0

    # ── Solver options ───────────────────────────────────────────────────
    ocp.solver_options.qp_solver       = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx  = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tol             = 1e-4
    ocp.solver_options.tf              = t_horizon

    return ocp


# ──────────────────────────────────────────────────────────────────────────────
#  Solver factory
# ──────────────────────────────────────────────────────────────────────────────

def build_mpcc_solver(x0, N_prediction, t_prediction, s_max=None,
                      use_cython=True):
    """Create, code-generate, compile and return the strict MPCC solver.

    Parameters
    ----------
    x0            : ndarray (14,)  – initial state [p,v,q,ω,θ₀]
    N_prediction  : int
    t_prediction  : float  [s]
    s_max         : float or None  – total arc length
    use_cython    : bool

    Returns
    -------
    acados_ocp_solver : AcadosOcpSolver
    ocp               : AcadosOcp
    model             : AcadosModel
    f_system          : casadi.Function(x14, u5) → ẋ14
    """
    ocp = create_mpcc_ocp_description(x0, N_prediction, t_prediction,
                                      s_max=s_max)
    model = ocp.model

    # Dynamics function for external RK4 simulation
    _, f_system, _, _ = f_system_model_mpcc()

    solver_json = 'acados_ocp_' + model.name + '.json'

    if use_cython:
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    else:
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, ocp, model, f_system
