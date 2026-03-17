"""
MPCC_baseline.py  –  Strict MPCC for quadrotor trajectory tracking.

Strict MPCC formulation:
  • θ (arc-length progress) is an optimisation STATE   (x[13])
  • v_θ (progress velocity) is an optimisation CONTROL (u[4])
  • Dynamics:  θ̇ = v_θ
  • The solver maximises v_θ to push the UAV along the path.
  • Reference points are evaluated at predicted θ values (per shooting node).

Uses the modular project structure:
    utils/   → quaternion & rotation helpers
    models/  → quadrotor dynamics (CasADi / acados)
    ocp/     → MPCC formulation & solver build
"""

import numpy as np
import time
import time as time_module
import math
import os
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import bisect
from scipy.io import savemat

# ── Project modules ──────────────────────────────────────────────────────────
from utils.quaternion_utils import euler_to_quaternion, Euler_p
from ocp.mpcc_controller import (
    build_mpcc_solver,
    create_casadi_position_interpolator,
    create_casadi_tangent_interpolator,
    create_casadi_quat_interpolator,
)
from graficas import (
    plot_pose, plot_error, plot_time, plot_control,
    plot_vel_lineal, plot_vel_angular, plot_CBF,
    plot_progress_velocity,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

VALUE  = 10
VALUEB = 7

UAV_R      = 0.15
MARGEN     = 0.1
OBSMOVIL_R = 0.4


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory generators
# ═══════════════════════════════════════════════════════════════════════════════

def trayectoria(t):
    v = VALUE
    xd   = lambda t: 7 * np.sin(v * 0.04 * t) + 3
    yd   = lambda t: 7 * np.sin(v * 0.08 * t)
    zd   = lambda t: 1.5 * np.sin(v * 0.08 * t) + 6
    xd_p = lambda t: 7 * v * 0.04 * np.cos(v * 0.04 * t)
    yd_p = lambda t: 7 * v * 0.08 * np.cos(v * 0.08 * t)
    zd_p = lambda t: 1.5 * v * 0.08 * np.cos(v * 0.08 * t)
    return xd, yd, zd, xd_p, yd_p, zd_p


def trayectoriaB(t):
    v = VALUEB
    xd   = lambda t: 7 * np.sin(-v * 0.04 * t) + 3
    yd   = lambda t: 7 * np.sin(-v * 0.08 * t)
    zd   = lambda t: 1.5 * np.sin(-v * 0.08 * t) + 6
    xd_p = lambda t: -7 * v * 0.04 * np.cos(-v * 0.04 * t)
    yd_p = lambda t: -7 * v * 0.08 * np.cos(-v * 0.08 * t)
    zd_p = lambda t: -1.5 * v * 0.08 * np.cos(-v * 0.08 * t)
    return xd, yd, zd, xd_p, yd_p, zd_p


# ═══════════════════════════════════════════════════════════════════════════════
#  Arc-length parameterisation
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):
    r       = lambda t: np.array([xd(t), yd(t), zd(t)])
    r_prime = lambda t: np.array([xd_p(t), yd_p(t), zd_p(t)])

    def integrand(t):
        return np.linalg.norm(r_prime(t))

    def arc_length(tk, t0=0):
        length, _ = quad(integrand, t0, tk, limit=100)
        return length

    positions   = []
    arc_lengths = []
    for tk in t_range:
        arc_lengths.append(arc_length(tk))
        positions.append(r(tk))

    arc_lengths = np.array(arc_lengths)
    positions   = np.array(positions).T  # (3, N)

    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])

    total_arc_length = arc_lengths[-1]

    def position_by_arc_length(s):
        """Evaluate path position at arc-length s (clamped to valid range)."""
        s = np.clip(s, arc_lengths[0], arc_lengths[-1])
        te = spline_t(s)
        return np.array([spline_x(te), spline_y(te), spline_z(te)])

    def tangent_by_arc_length(s, ds=1e-4):
        """Evaluate unit tangent at arc-length s via finite differences."""
        s = np.clip(s, arc_lengths[0], arc_lengths[-1])
        s_fwd = np.clip(s + ds, arc_lengths[0], arc_lengths[-1])
        s_bwd = np.clip(s - ds, arc_lengths[0], arc_lengths[-1])
        p_fwd = position_by_arc_length(s_fwd)
        p_bwd = position_by_arc_length(s_bwd)
        tang  = (p_fwd - p_bwd) / (s_fwd - s_bwd + 1e-10)
        norm  = np.linalg.norm(tang)
        if norm > 1e-8:
            tang /= norm
        return tang

    return arc_lengths, positions, position_by_arc_length, tangent_by_arc_length, total_arc_length


# ═══════════════════════════════════════════════════════════════════════════════
#  MPCC error computation (NumPy)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_errors_norm(sd, sd_p, model_x):
    """Compute contouring / lag errors, total error, and progress velocity."""
    e_t = sd - model_x[0:3]
    tangent = sd_p

    # Lag error
    el = np.dot(tangent, e_t) * tangent

    # Contouring error
    I = np.eye(3)
    P_ec = I - np.outer(tangent, tangent)
    ec   = P_ec @ e_t

    error_total = ec + el

    return ec, el, error_total


# ──────────────────────────────────────────────────────────────────────────────
#  RK4 integrator for the augmented system (14 states, 5 controls)
# ──────────────────────────────────────────────────────────────────────────────

def f_d_mpcc(x, u, ts, f_sys):
    """One-step RK4 integration for the augmented MPCC system.

    Parameters
    ----------
    x     : ndarray (14,)  – [p, v, q, ω, θ]
    u     : ndarray (5,)   – [T, τx, τy, τz, v_θ]
    ts    : float          – sampling time [s]
    f_sys : casadi.Function(x14, u5) → ẋ14

    Returns
    -------
    x_next : ndarray (14,)
    """
    k1 = f_sys(x, u)
    k2 = f_sys(x + (ts / 2) * k1, u)
    k3 = f_sys(x + (ts / 2) * k2, u)
    k4 = f_sys(x + ts * k3, u)
    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array(x_next[:, 0]).reshape((14,))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Timing configuration ─────────────────────────────────────────────
    t_final = 30                                       # [s]
    frec    = 100                                       # [Hz]
    t_s     = 1 / frec                                 # [s]
    t_prediction = 0.3                                 # [s]
    N_prediction = int(round(t_prediction / t_s))      # auto
    print(f"[CONFIG]  frec={frec} Hz  |  t_s={t_s*1e3:.2f} ms  "
          f"|  t_prediction={t_prediction} s  |  N_prediction={N_prediction} steps")

    # ── Time vector ──────────────────────────────────────────────────────
    t = np.arange(0, t_final + t_s, t_s)
    N_sim = t.shape[0] - N_prediction

    # ── Trajectory A (UAV path) ──────────────────────────────────────────
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)
    xd_obs1, yd_obs1, zd_obs1, _, _, _ = trayectoriaB(t)

    # Arc-length parameterisation  (now also returns tangent_by_arc_length)
    t_finer = np.linspace(0, t_final, len(t))
    arc_lengths, pos_ref, position_by_arc_length, tangent_by_arc_length, \
        total_arc_length = \
        calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p,
                                           t_finer, t_max=t_final)
    s_max = total_arc_length
    print(f"[ARC]  Total arc length = {s_max:.3f} m")

    # ── Storage vectors ──────────────────────────────────────────────────
    delta_t        = np.zeros((1, N_sim), dtype=np.double)
    h_CBF_1        = np.zeros((1, N_sim), dtype=np.double)
    h_CBF_2        = np.zeros((1, N_sim), dtype=np.double)
    CLF            = np.zeros((1, N_sim), dtype=np.double)
    e_contorno     = np.zeros((3, N_sim), dtype=np.double)
    e_arrastre     = np.zeros((3, N_sim), dtype=np.double)
    e_total        = np.zeros((3, N_sim), dtype=np.double)
    vel_progres    = np.zeros((1, N_sim), dtype=np.double)     # v_θ (from solver)
    vel_real       = np.zeros((1, N_sim), dtype=np.double)     # real progress speed (dot(tangent, v))
    theta_history  = np.zeros((1, N_sim + 1), dtype=np.double) # θ state
    t_solver       = np.zeros((1, N_sim), dtype=np.double)
    t_loop         = np.zeros((1, N_sim), dtype=np.double)

    # ── Initial state (14-dim: [p, v, q, ω, θ₀]) ───────────────────────
    #    Start at θ₀ = 0 (beginning of the path)
    x = np.zeros((14, N_sim + 1), dtype=np.double)
    x[:, 0] = [5, 0.0, 7,                       # position
               0.0, 0.0, 0.0,                   # velocity
               1, 0, 0, 0,                       # quaternion (identity)
               0.0, 0.0, 0.0,                    # angular velocity
               0.0]                               # θ₀ = 0
    theta_history[0, 0] = x[13, 0]

    # ── Desired yaw from tangent (precomputed for reference quaternions) ─
    xd_p_vals = xd_p(t)
    yd_p_vals = yd_p(t)
    psid = np.arctan2(yd_p_vals, xd_p_vals)

    quatd = np.zeros((4, t.shape[0]), dtype=np.double)
    for i in range(t.shape[0]):
        quatd[:, i] = euler_to_quaternion(0, 0, psid[i])

    # ── Reference for plotting (17-dim, time-indexed) ────────────────────
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    xref = np.zeros((17, t.shape[0]), dtype=np.double)
    xref[0, :]  = pos_ref[0, :]   # px_d
    xref[1, :]  = pos_ref[1, :]   # py_d
    xref[2, :]  = pos_ref[2, :]   # pz_d
    xref[3, :]  = dp_ds[0, :]     # tangent_x
    xref[4, :]  = dp_ds[1, :]     # tangent_y
    xref[5, :]  = dp_ds[2, :]     # tangent_z
    xref[6, :]  = quatd[0, :]     # qw_d
    xref[7, :]  = quatd[1, :]     # qx_d
    xref[8, :]  = quatd[2, :]     # qy_d
    xref[9, :]  = quatd[3, :]     # qz_d

    # ── Mobile obstacle trajectory ───────────────────────────────────────
    movil_obs1 = np.zeros((3, t.shape[0]), dtype=np.double)
    movil_obs1[0, :] = xd_obs1(t)
    movil_obs1[1, :] = yd_obs1(t)
    movil_obs1[2, :] = zd_obs1(t)

    # ── Control storage (5-dim: T, τx, τy, τz, v_θ) ─────────────────────
    u_control = np.zeros((5, N_sim), dtype=np.double)

    # ── Create CasADi trajectory interpolation  (θ → reference) ──────────
    #    Discretise the arc-length into waypoints, then build piecewise-linear
    #    CasADi functions so the solver can differentiate through θ.
    N_WAYPOINTS = 30                # 200 → max error < 6 cm, ~10 ms solver
    s_wp = np.linspace(0, s_max, N_WAYPOINTS)
    pos_wp  = np.zeros((3, N_WAYPOINTS))
    tang_wp = np.zeros((3, N_WAYPOINTS))
    quat_wp = np.zeros((4, N_WAYPOINTS))

    for i, sv in enumerate(s_wp):
        pos_wp[:, i]  = position_by_arc_length(sv)
        tang_wp[:, i] = tangent_by_arc_length(sv)
        # Quaternion from yaw of tangent
        psi_i = np.arctan2(tang_wp[1, i], tang_wp[0, i])
        quat_wp[:, i] = euler_to_quaternion(0, 0, psi_i)

    # Correct quaternion hemisphere (keep dot(q_i, q_{i-1}) > 0)
    for i in range(1, N_WAYPOINTS):
        if np.dot(quat_wp[:, i], quat_wp[:, i - 1]) < 0:
            quat_wp[:, i] *= -1

    gamma_pos  = create_casadi_position_interpolator(s_wp, pos_wp)
    gamma_vel  = create_casadi_tangent_interpolator(s_wp, tang_wp)
    gamma_quat = create_casadi_quat_interpolator(s_wp, quat_wp)
    print(f"[INTERP] Created CasADi interpolation with {N_WAYPOINTS} waypoints")

    # ── Build MPCC solver ────────────────────────────────────────────────
    acados_ocp_solver, ocp, model, f = build_mpcc_solver(
        x[:, 0], N_prediction, t_prediction, s_max=s_max,
        gamma_pos=gamma_pos, gamma_vel=gamma_vel, gamma_quat=gamma_quat,
        use_cython=True,
    )

    nx = model.x.size()[0]   # 14
    nu = model.u.size()[0]   # 5

    simX = np.ndarray((nx, N_prediction + 1))
    simU = np.ndarray((nu, N_prediction))

    # Warm-start
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    print("Initializing simulation...")

    # ── Static obstacle positions ────────────────────────────────────────
    obs_pos = np.array([
        [4.8753,  3.6136, 6.7743],
        [9.8215,  3.0610, 6.6559],
        [9.9405, -1.8058, 5.6130],
        [5.0403, -3.9034, 5.1636],
        [0.5521,  4.5867, 6.9829],
        [-3.8693, 2.6429, 6.5663],
        [-3.7215,-3.7539, 5.1956],
        [0.8770, -4.0461, 5.1330],
    ])
    obs_ra = 0.8 * np.array([0.35, 0.40, 0.45, 0.5, 0.45, 0.35, 0.30, 0.25])

    # ══════════════════════════════════════════════════════════════════════
    #  Control loop  (strict MPCC: reference computed from predicted θ)
    # ══════════════════════════════════════════════════════════════════════
    print("Ready!!!")
    time_all = time.time()

    for k in range(N_sim):
        tic = time.time()

        # ── Stop when path is complete ────────────────────────────────────
        if x[13, k] >= s_max - 0.01:
            print(f"[k={k:04d}]  Path complete at θ={x[13,k]:.3f} m. Stopping.")
            N_sim = k   # trim storage arrays to actual run length
            break

        # ── Find closest obstacle ────────────────────────────────────────
        distances   = np.linalg.norm(obs_pos - x[0:3, k], axis=1)
        idx_closest = np.argmin(distances)

        obs_x_c = obs_pos[idx_closest, 0]
        obs_y_c = obs_pos[idx_closest, 1]
        obs_z_c = obs_pos[idx_closest, 2]
        obs_r_c = obs_ra[idx_closest]

        # ── Set initial state (14-dim) ───────────────────────────────────
        acados_ocp_solver.set(0, "lbx", x[:, k])
        acados_ocp_solver.set(0, "ubx", x[:, k])

        # ── Evaluate CBF values ──────────────────────────────────────────
        obst_static = np.array([obs_x_c, obs_y_c, obs_z_c])
        obst_movil  = movil_obs1[:3, k]

        h_CBF_1[:, k] = np.linalg.norm(x[:3, k] - obst_static) - (UAV_R + obs_r_c + MARGEN)
        h_CBF_2[:, k] = np.linalg.norm(x[:3, k] - obst_movil)  - (UAV_R + OBSMOVIL_R + MARGEN)

        # Obstacle values vector (7 elements)
        values = [obs_x_c, obs_y_c, obs_z_c, obs_r_c,
                  movil_obs1[0, k], movil_obs1[1, k], movil_obs1[2, k]]

        # ── Set parameters per shooting node (obstacles only, 7-dim) ──────
        # Reference is now computed SYMBOLICALLY inside the OCP from θ,
        # so we only need to pass obstacle data as parameters.
        p_val = np.array(values)                            # 7-dim
        for j in range(N_prediction + 1):
            acados_ocp_solver.set(j, "p", p_val)

        # ── Solve ────────────────────────────────────────────────────────
        tic_solver = time.time()
        status = acados_ocp_solver.solve()
        t_solver[:, k] = time.time() - tic_solver

        # ── Read predicted trajectory (for next iteration warm-start) ────
        for i in range(N_prediction):
            simX[:, i] = acados_ocp_solver.get(i, "x")
            simU[:, i] = acados_ocp_solver.get(i, "u")
        simX[:, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        # ── Get optimal control (5-dim) ──────────────────────────────────
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        # ── Store v_θ and θ ──────────────────────────────────────────────
        vel_progres[:, k]       = u_control[4, k]    # v_θ (solver input)
        theta_history[:, k]     = x[13, k]            # current θ

        # Real progress speed = projection of UAV velocity onto tangent
        theta_k_now = np.clip(x[13, k], 0.0, s_max)
        tang_k_now  = tangent_by_arc_length(theta_k_now)
        vel_real[:, k] = np.dot(tang_k_now, x[3:6, k])

        # ── System evolution (augmented RK4, 14 states) ──────────────────
        x[:, k + 1] = f_d_mpcc(x[:, k], u_control[:, k], t_s, f)

        # Clamp θ to valid range
        x[13, k + 1] = np.clip(x[13, k + 1], 0.0, s_max)
        theta_history[:, k + 1] = x[13, k + 1]

        # ── Rate control — run in real time ──────────────────────────────
        elapsed = time.time() - tic
        remaining = t_s - elapsed
        if remaining > 0:
            time_module.sleep(remaining)

        t_loop[:, k] = time.time() - tic
        delta_t[:, k] = t_loop[:, k]

        # ── Compute MPCC errors (using current θ-based reference) ────────
        theta_k = x[13, k]
        sd_k    = position_by_arc_length(theta_k)
        tang_k  = tangent_by_arc_length(theta_k)
        e_contorno[:, k], e_arrastre[:, k], e_total[:, k] = \
            calculate_errors_norm(sd_k, tang_k, x[0:6, k])

        # ── Print progress ───────────────────────────────────────────────
        overrun = " ⚠ OVERRUN" if elapsed > t_s else ""
        ratio_vtheta = vel_real[0,k] / (vel_progres[0,k] + 1e-8)
        print(f"[k={k:04d}]  solver={t_solver[0,k]*1e3:5.2f} ms  "
              f"|  v_θ={vel_progres[0,k]:5.2f}  v_real={vel_real[0,k]:5.2f}  "
              f"ratio={ratio_vtheta:4.2f}  |  "
              f"θ={x[13,k]:7.2f}/{s_max:.0f} m  |  "
              f"{1/t_loop[0,k]:5.1f} Hz{overrun}")

    # ══════════════════════════════════════════════════════════════════════
    #  Post-processing  (trim all arrays to the actual run length N_sim)
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - time_all
    print(f"\nTiempo total de ejecución: {total_time:.4f} segundos")
    print(f"Final θ = {x[13, N_sim]:.3f} / {s_max:.3f} m  "
          f"({x[13, N_sim]/s_max*100:.1f}% of path)")

    # Trim all storage arrays to actual run length
    x            = x[:, :N_sim + 1]
    u_control    = u_control[:, :N_sim]
    h_CBF_1      = h_CBF_1[:, :N_sim]
    h_CBF_2      = h_CBF_2[:, :N_sim]
    CLF          = CLF[:, :N_sim]
    e_contorno   = e_contorno[:, :N_sim]
    e_arrastre   = e_arrastre[:, :N_sim]
    e_total      = e_total[:, :N_sim]
    vel_progres  = vel_progres[:, :N_sim]
    vel_real     = vel_real[:, :N_sim]
    theta_history= theta_history[:, :N_sim + 1]
    t_solver     = t_solver[:, :N_sim]
    t_loop       = t_loop[:, :N_sim]
    t_plot       = t[:N_sim + 1]

    print("Generating figures...")

    # Build reference from θ history for plotting
    xref_theta = np.zeros((17, N_sim + 1))
    for i in range(N_sim + 1):
        s_i = theta_history[0, i]
        pos_i  = position_by_arc_length(s_i)
        tang_i = tangent_by_arc_length(s_i)
        xref_theta[0:3, i]  = pos_i
        xref_theta[3:6, i]  = tang_i

    fig1 = plot_pose(x[:13, :], xref_theta, t_plot)
    fig1.savefig("1_pose.png");   print("✓ Saved 1_pose.png")

    fig3 = plot_vel_lineal(x[3:6, :], t_plot)
    fig3.savefig("3_vel_lineal.png");  print("✓ Saved 3_vel_lineal.png")

    fig4 = plot_vel_angular(x[10:13, :], t_plot)
    fig4.savefig("4_vel_angular.png"); print("✓ Saved 4_vel_angular.png")

    fig5 = plot_CBF(h_CBF_1, t_plot[:N_sim])
    fig5.savefig("5_CBF.png");  print("✓ Saved 5_CBF.png")

    fig6 = plot_CBF(CLF, t_plot[:N_sim])
    fig6.savefig("7_CLF.png");  print("✓ Saved 7_CLF.png")

    # Plot first 4 control inputs (thrust + torques)
    fig2 = plot_control(u_control[:4, :], t_plot[:N_sim])
    fig2.savefig("2_control_actions.png"); print("✓ Saved 2_control_actions.png")

    # Plot v_θ vs v_real and θ progress
    fig_vprog = plot_progress_velocity(vel_progres, vel_real, theta_history, t_plot[:N_sim])
    fig_vprog.savefig("8_progress_velocity.png", dpi=150)
    print("✓ Saved 8_progress_velocity.png")

    # ── Timing statistics ────────────────────────────────────────────────
    s_ms  = t_solver[0, :] * 1e3
    l_ms  = t_loop[0, :]   * 1e3
    ts_ms = t_s * 1e3
    n_overrun = int(np.sum(l_ms > ts_ms * 1.05))

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print(  "║                     TIMING STATISTICS                          ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Nominal t_s = {ts_ms:5.2f} ms  ({frec:.0f} Hz)                              ║")
    print(  "╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  [Solver]  mean={np.mean(s_ms):5.2f}  max={np.max(s_ms):5.2f}  std={np.std(s_ms):4.2f}  ms     ║")
    print(f"║  [Loop  ]  mean={np.mean(l_ms):5.2f}  max={np.max(l_ms):5.2f}  std={np.std(l_ms):4.2f}  ms     ║")
    print(f"║  Freq real : {1000/np.mean(l_ms):5.1f} Hz                                  ║")
    print(f"║  Overruns  : {n_overrun:4d} / {len(l_ms)} iters ({n_overrun/len(l_ms)*100:.1f} %)                 ║")
    print(  "╚══════════════════════════════════════════════════════════════════╝\n")

    # ── v_θ and θ statistics ─────────────────────────────────────────────
    print(f"[v_θ]  mean={np.mean(vel_progres):6.3f}  max={np.max(vel_progres):6.3f}  "
          f"min={np.min(vel_progres):6.3f}")
    print(f"[v_r]  mean={np.mean(vel_real):6.3f}  max={np.max(vel_real):6.3f}  "
          f"min={np.min(vel_real):6.3f}")
    # Only compute ratio where v_θ > 0 to avoid divide-by-near-zero noise
    mask = vel_progres[0, :] > 0.1
    if np.any(mask):
        ratio_mean = np.mean(vel_real[0, mask] / vel_progres[0, mask])
        print(f"[ratio v_real/v_θ]  mean={ratio_mean:5.3f}  (only where v_θ>0.1)")
    print(f"[θ  ]  final={x[13, N_sim]:8.3f}  /  {s_max:.3f} m  "
          f"({x[13, N_sim]/s_max*100:.1f}%)\n")

    # ── Save results ─────────────────────────────────────────────────────
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Results_MPCC_CLF_CBF"
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Usando directorio local.")
        pwd = os.getcwd()

    experiment_number = 1
    name_file = f"Results_static_MPCC_CLF_CBF_{experiment_number}.mat"

    savemat(os.path.join(pwd, name_file), {
        'states': x,
        'T_control': u_control,
        'CBF_1': h_CBF_1,
        'CBF_2': h_CBF_2,
        'CLF': CLF,
        'time': t_plot,
        'ref': xref[:, :N_sim + 1],
        'obs_movil': movil_obs1[:, :N_sim],
        'e_total': e_total,
        'e_contorno': e_contorno,
        'e_arrastre': e_arrastre,
        'vel_progres': vel_progres,
        'vel_real': vel_real,
        'theta_history': theta_history,
    })
    print(f"✓ Results saved to {os.path.join(pwd, name_file)}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
    else:
        print("Complete Execution")
