from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from ode_acados import (quadrotorModel, left_jacobian_SO3, left_jacobian_SO3_inv,
                        quaternion_to_rotation_matrix, extract_log_error_components,
                        compute_rho_translational_error, logarithmic_lag_contouring_components,
                        lie_invariant_projection)
from casadi import Function, MX, vertcat, sin, cos, fabs, DM
import casadi as ca

import numpy as np
from casadi import sqrt, dot, fabs, sign


def dual_quaternion_to_position(dq):
    r = dq[0:4]
    d = dq[4:8]
    r_conj = ca.vertcat(r[0], -r[1], -r[2], -r[3])

    def quat_mul(q1, q2):
        s1, v1 = q1[0], q1[1:4]
        s2, v2 = q2[0], q2[1:4]
        scalar = s1 * s2 - ca.dot(v1, v2)
        vector = s1 * v2 + s2 * v1 + ca.cross(v1, v2)
        return ca.vertcat(scalar, vector)

    p_quat = 2 * quat_mul(d, r_conj)
    position = p_quat[1:4]
    return position


def create_ocp_solver(
    x0,
    N_horizon,
    t_horizon,
    F_max,
    F_min,
    tau_1_max,
    tau_1_min,
    tau_2_max,
    tau_2_min,
    tau_3_max,
    tau_3_min,
    L,
    ts,
    u_s_min=0.2,
    u_s_max=5.0,
    gamma_dq=None,
    gamma_vel=None,
) -> AcadosOcp:

    ocp = AcadosOcp()

    model, get_trans, get_quat, constraint, error_lie_2, dual_error, ln, Ad, conjugate, rotation = quadrotorModel(L)

    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    ocp.p = model.p
    ocp.dims.N = N_horizon

    R = MX.zeros(4, 4)
    R[0, 0] = 20/F_max
    R[1, 1] = 60/tau_1_max
    R[2, 2] = 60/tau_2_max
    R[3, 3] = 60/tau_3_max

    Q_s = 0.3

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    s_state = model.x[14]

    # Si gamma_dq está definido, usar interpolación CasADi (referencia dinámica basada en s)
    # Si no, usar ocp.p (referencia pasada como parámetro en cada iteración)
    if gamma_dq is not None:
        dual_d = gamma_dq(s_state)
        sd_p   = gamma_vel(s_state) if gamma_vel is not None else ocp.p[11:14]
    else:
        dual_d = ocp.p[0:8]
        sd_p   = ocp.p[11:14]

    dual = model.x[0:8]

    error_total_lie = error_lie_2(dual_d, dual)
    error    = dual_error(dual_d, dual)
    error_c  = conjugate(error)
    ln_error = ln(error)

    nominal_input       = ocp.p[14:18]
    error_nominal_input = nominal_input - model.u[0:4]

    w_b = model.x[8:11]
    v_b = model.x[11:14]
    v_i = rotation(model.x[0:4], v_b)

    w_b_d   = ocp.p[8:11]
    v_i_d   = ocp.p[11:14]
    error_w = w_b - w_b_d
    error_v = v_i - v_i_d

    Q_l = MX.zeros(6, 6)
    Q_l[0, 0] = 10
    Q_l[1, 1] = 10
    Q_l[2, 2] = 10
    Q_l[3, 3] = 0
    Q_l[4, 4] = 0
    Q_l[5, 5] = 0

    phi            = ln_error[0:3]
    Jl             = left_jacobian_SO3(phi)
    s_pos          = dual_quaternion_to_position(dual)
    sd_pos         = dual_quaternion_to_position(dual_d)
    q_desired_real = dual_d[0:4]
    rho            = compute_rho_translational_error(phi, q_desired_real, s_pos, sd_pos)

    lambda_weight = 1.0
    mu_weight     = 1.0
    rho_l, rho_c, rho_magnitude = lie_invariant_projection(phi, rho, sd_p, lambda_weight, mu_weight)

    # ── Pesos del costo ───────────────────────────────────────────────────
    Q_el   = 5.0  * np.eye(3)
    Q_ec   = 10.0 * np.eye(3)
    U_mat  = np.diag([0.1, 250, 250, 250])
    Q_omega = 0.5

    omega_cost         = Q_omega * (w_b.T @ w_b)
    control_cost       = model.u[0:4].T @ U_mat @ model.u[0:4]
    error_lag_log      = rho_l.T @ Q_el @ rho_l
    error_contorno_log = rho_c.T @ Q_ec @ rho_c

    u_s               = model.u[4]
    arc_speed_penalty = Q_s * (u_s_max - u_s)**2

    ocp.model.cost_expr_ext_cost = (10*(ln_error.T @ Q_l @ ln_error) +
                                    control_cost +
                                    error_contorno_log +
                                    error_lag_log +
                                    arc_speed_penalty +
                                    omega_cost)

    ocp.model.cost_expr_ext_cost_e = (10*(ln_error.T @ Q_l @ ln_error) +
                                      error_contorno_log +
                                      error_lag_log +
                                      omega_cost)

    # ── Parámetros y restricciones ────────────────────────────────────────
    ocp.parameter_values = np.zeros(18)

    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu   = np.array([F_min, tau_1_min, tau_2_min, tau_3_min, u_s_min])
    ocp.constraints.ubu   = np.array([F_max, tau_1_max, tau_2_max, tau_3_max, u_s_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
    ocp.constraints.x0    = x0

    ocp.model.con_h_expr = constraint.expr
    nsbx = 0
    nh   = constraint.expr.shape[0]
    nsh  = nh
    ns   = nsh + nsbx

    ocp.cost.zl = 100 * np.ones((ns, ))
    ocp.cost.Zl = 100 * np.ones((ns, ))
    ocp.cost.Zu = 100 * np.ones((ns, ))
    ocp.cost.zu = 100 * np.ones((ns, ))

    ocp.constraints.lh    = np.array([constraint.min])
    ocp.constraints.uh    = np.array([constraint.max])
    ocp.constraints.lsh   = np.zeros(nsh)
    ocp.constraints.ush   = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))

    # ── Opciones del solver ───────────────────────────────────────────────
    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N_horizon // 4
    ocp.solver_options.hessian_approx   = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.integrator_type  = "ERK"
    ocp.solver_options.nlp_solver_type  = "SQP_RTI"
    ocp.solver_options.Tsim             = ts
    ocp.solver_options.tf               = t_horizon
    return ocp
