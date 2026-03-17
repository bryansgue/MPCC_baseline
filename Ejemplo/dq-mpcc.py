#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import itertools
from plotting_utils import (
    plot_states_quaternion, plot_states_position, fancy_plots_4, fancy_plots_3,
    plot_control_actions, fancy_plots_1, plot_cost_total,
    plot_angular_velocities, plot_linear_velocities,
    plot_cost_orientation, plot_cost_translation, plot_cost_control,
    plot_norm_quat, plot_norm_real, plot_norm_dual,
    plot_lyapunov_dot, plot_lyapunov, plot_time,
    plot_curvature_vs_velocity,
    plot_trajectory_3d, plot_trajectory_xy, plot_trajectory_xz,
)
from nav_msgs.msg import Odometry
from functions import dualquat_from_pose_casadi
from ode_acados import dualquat_trans_casadi, dualquat_quat_casadi, rotation_casadi, rotation_inverse_casadi, dual_velocity_casadi, dual_quat_casadi, velocities_from_twist_casadi
from ode_acados import f_rk4_casadi_simple, noise, cost_quaternion_casadi, cost_translation_casadi
from ode_acados import error_dual_aux_casadi
from ode_acados import create_casadi_trajectory_interpolator, create_casadi_dual_quaternion_interpolator
from nmpc_acados import create_ocp_solver
from ode_acados import compute_flatness_states, trajectory
from acados_template import AcadosOcpSolver, AcadosSimSolver
import scipy.io
from scipy.io import savemat
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from scipy.integrate import quad
from scipy.optimize import bisect
import math
import time
from scipy.interpolate import CubicSpline

# Creating Funtions based on Casadi
dualquat_from_pose = dualquat_from_pose_casadi()
get_trans = dualquat_trans_casadi()
get_quat = dualquat_quat_casadi()
dual_twist = dual_velocity_casadi()
velocity_from_twist = velocities_from_twist_casadi()
rot = rotation_casadi()
inverse_rot = rotation_inverse_casadi()
f_rk4 = f_rk4_casadi_simple()
cost_quaternion = cost_quaternion_casadi()
cost_translation = cost_translation_casadi()
error_dual_f = error_dual_aux_casadi()

file_path_1 = "/home/uav/catkin_ws/src/dual_quaternion/scripts/MPC_LIE_FINAL/Separed_cost.mat"
Identification = scipy.io.loadmat(file_path_1) 
#Identification = scipy.io.loadmat('Separed_cost_without_velocities.mat') 
x_0 = Identification['x_init']

import os
script_dir = os.path.dirname(__file__)
#folder_path = os.path.join(script_dir, 'cost_with_velocities/')
folder_path = script_dir

# Definir el valor global
value = 10

def get_odometry(odom_msg, dqd, name):
    # Function to send the Oritentation of the Quaternion
    # Get Information from the DualQuaternion
    t_d = get_trans(dqd)

    q_d = get_quat(dqd)

    odom_msg.header.stamp = rospy.Time.now()
    odom_msg.header.frame_id = "world"
    odom_msg.child_frame_id = name
    odom_msg.pose.pose.position.x = t_d[1]
    odom_msg.pose.pose.position.y = t_d[2]
    odom_msg.pose.pose.position.z = t_d[3]

    odom_msg.pose.pose.orientation.x = q_d[1]
    odom_msg.pose.pose.orientation.y = q_d[2]
    odom_msg.pose.pose.orientation.z = q_d[3]
    odom_msg.pose.pose.orientation.w = q_d[0]
    return odom_msg

def send_odometry(odom_msg, odom_pub):
    # Function to send the orientation of the Quaternion
    odom_pub.publish(odom_msg)
    return None

def init_marker(marker_msg, x):
    marker_msg.header.frame_id = "world"
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = "trajectory"
    marker_msg.id = 0
    marker_msg.type = Marker.LINE_STRIP
    marker_msg.action = Marker.ADD
    marker_msg.pose.orientation.w = 1.0
    marker_msg.scale.x = 0.02  # Line width
    marker_msg.color.a = 1.0  # Alpha
    marker_msg.color.r = 0.0  # Red
    marker_msg.color.g = 1.0  # Green
    marker_msg.color.b = 0.0  # Blue
    point = Point()
    point.x = x[0]
    point.y = x[1]
    point.z = x[2]
    points = [point]
    marker_msg.points = points
    return marker_msg, points

def send_marker(marker_msg, points, publisher, x):
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.type = Marker.LINE_STRIP
    marker_msg.action = Marker.ADD
    point = Point()
    point.x = x[0]
    point.y = x[1]
    point.z = x[2]
    points.append(point)
    marker_msg.points = points
    publisher.publish(marker_msg)
    return marker_msg, points

def trayectoria(t):

    def xd(t):
        return 7 * np.sin(value * 0.04 * t) + 3

    def yd(t):
        return 7 * np.sin(value * 0.08 * t)

    def zd(t):
        return 1.5 * np.sin(value * 0.08 * t) + 6

    def xd_p(t):
        return 7 * value * 0.04 * np.cos(value * 0.04 * t)

    def yd_p(t):
        return 7 * value * 0.08 * np.cos(value * 0.08 * t)

    def zd_p(t):
        return 1.5 * value * 0.08 * np.cos(value * 0.08 * t)

    return xd, yd, zd, xd_p, yd_p, zd_p


def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]

def calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):

    
    def r(t):
        """ Devuelve el punto en la trayectoria para el parámetro t usando las funciones de trayectoria. """
        return np.array([xd(t), yd(t), zd(t)])

    def r_prime(t):
        """ Devuelve la derivada de la trayectoria en el parámetro t usando las derivadas de las funciones de trayectoria. """
        return np.array([xd_p(t), yd_p(t), zd_p(t)])

    def integrand(t):
        """ Devuelve la norma de la derivada de la trayectoria en el parámetro t. """
        return np.linalg.norm(r_prime(t))

    def arc_length(tk, t0=0):
        """ Calcula la longitud de arco desde t0 hasta tk usando las derivadas de la trayectoria. """
        length1, _ = quad(integrand, t0, (t0 + tk) / 2, limit=50)
        length2, _ = quad(integrand, (t0 + tk) / 2, tk, limit=50)
        length = length1 + length2
        length, _ = quad(integrand, t0, tk, limit=100)
        return length

    def find_t_for_length(theta, t0=0):
        """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
        func = lambda t: arc_length(t, t0) - theta
        return bisect(func, t0, t_max)

    # Generar las posiciones y longitudes de arco
    positions = []
    arc_lengths = []
    
    for tk in t_range:
        theta = arc_length(tk)
        arc_lengths.append(theta)
        point = r(tk)
        positions.append(point)

    arc_lengths = np.array(arc_lengths)
    positions = np.array(positions).T  # Convertir a array 2D (3, N)

    # Crear splines cúbicos para la longitud de arco con respecto al tiempo
    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])
    
    # ============ CORRECCIÓN DE ÁNGULO YAW (evitar discontinuidad en ±π) ============
    # Calcular ángulo yaw para cada punto de tiempo y aplicar unwrap
    yaw_angles = []
    for tk in t_range:
        yaw = np.arctan2(yd_p(tk), xd_p(tk))
        yaw_angles.append(yaw)
    yaw_angles = np.array(yaw_angles)
    # np.unwrap corrige los saltos de ±2π para obtener ángulo continuo
    yaw_unwrapped = np.unwrap(yaw_angles)
    # Crear spline del ángulo continuo
    spline_yaw = CubicSpline(t_range, yaw_unwrapped)

    # Función que retorna la posición dado un valor de longitud de arco
    def position_by_arc_length(s):
        t_estimated = spline_t(s)  # Usar spline para obtener la estimación precisa de t
        return np.array([spline_x(t_estimated), spline_y(t_estimated), spline_z(t_estimated)])
    
    # NEW: Helper functions to get references by arc length
    def quaternion_by_arc_length(s):
        """Get quaternion at arc length s - usando ángulo unwrapped para evitar saltos"""
        t_estimated = spline_t(s)
        # Usar el spline del ángulo unwrapped (continuo)
        psid_val = spline_yaw(t_estimated)
        quat_val = euler_to_quaternion(0, 0, psid_val)
        return np.array(quat_val)
    
    def velocity_by_arc_length(s):
        """Get velocity direction at arc length s"""
        t_estimated = spline_t(s)
        return np.array([xd_p(t_estimated), yd_p(t_estimated), zd_p(t_estimated)])

    return arc_lengths, positions, position_by_arc_length, quaternion_by_arc_length, velocity_by_arc_length, spline_t, xd_p, yd_p, zd_p


def create_mpcc_waypoints(position_by_arc_length, velocity_by_arc_length, 
                          quaternion_by_arc_length, target_path_length, n_waypoints=30,
                          max_angle_deg=25):
    """
    Crea waypoints discretos para interpolación CasADi del MPCC.
    
    Esta función es NECESARIA porque CasADi necesita expresiones simbólicas para el optimizador.
    No puede llamar directamente a funciones Python arbitrarias durante la optimización.
    
    NUEVO: Detección automática de giros y refinamiento adaptivo.
    Si el cambio de ángulo entre waypoints consecutivos excede max_angle_deg,
    se insertan waypoints intermedios hasta que el cambio sea aceptable.
    
    Args:
        position_by_arc_length: Función que retorna posición [x, y, z] dado s
        velocity_by_arc_length: Función que retorna velocidad [vx, vy, vz] dado s
        quaternion_by_arc_length: Función que retorna quaternion [qw, qx, qy, qz] dado s
        target_path_length: Longitud total de la trayectoria [m]
        n_waypoints: Número base de waypoints (se añaden más si hay giros)
        max_angle_deg: Máximo cambio angular permitido entre waypoints [°]
    
    Returns:
        s_waypoints: Array de longitudes de arco [N]
        pos_waypoints: Posiciones en waypoints [3, N]
        vel_waypoints: Velocidades normalizadas en waypoints [3, N]
        quat_waypoints: Quaterniones con corrección de hemisferio [4, N]
    """
    
    # ============ PASO 1: MUESTREO INICIAL UNIFORME ============
    s_list = list(np.linspace(0, target_path_length, n_waypoints))
    
    # ============ PASO 2: EVALUAR QUATERNIONES Y DETECTAR GIROS ============
    def get_quat(s):
        q = quaternion_by_arc_length(np.clip(s, 0, target_path_length))
        return q / (np.linalg.norm(q) + 1e-10)
    
    def angle_between_quats(q1, q2):
        """Ángulo en grados entre dos quaterniones (usando camino corto)."""
        dot = abs(np.dot(q1, q2))
        return 2 * np.arccos(np.clip(dot, -1, 1)) * 180 / np.pi
    
    # ============ PASO 3: REFINAMIENTO ADAPTIVO ============
    # Iterar hasta que todos los ángulos sean menores que max_angle_deg
    max_iterations = 10  # Límite de seguridad
    for iteration in range(max_iterations):
        refined = False
        i = 0
        while i < len(s_list) - 1:
            s1, s2 = s_list[i], s_list[i+1]
            q1, q2 = get_quat(s1), get_quat(s2)
            angle = angle_between_quats(q1, q2)
            
            if angle > max_angle_deg and (s2 - s1) > 0.05:  # No subdividir si el segmento es muy corto
                # Insertar punto medio
                s_mid = (s1 + s2) / 2
                s_list.insert(i + 1, s_mid)
                refined = True
                # No incrementar i para revisar el nuevo segmento
            else:
                i += 1
        
        if not refined:
            break
    
    s_waypoints = np.array(s_list)
    n_final = len(s_waypoints)
    
    # ============ PASO 4: EVALUAR EN TODOS LOS WAYPOINTS ============
    pos_waypoints = np.zeros((3, n_final))
    vel_waypoints = np.zeros((3, n_final))
    quat_waypoints = np.zeros((4, n_final))
    
    for i, s_val in enumerate(s_waypoints):
        pos_waypoints[:, i] = position_by_arc_length(s_val)
        vel_waypoints[:, i] = velocity_by_arc_length(s_val)
        quat_waypoints[:, i] = get_quat(s_val)
    
    # ============ PASO 5: CORRECCIÓN DE HEMISFERIO ============
    for i in range(1, n_final):
        if np.dot(quat_waypoints[:, i-1], quat_waypoints[:, i]) < 0:
            quat_waypoints[:, i] = -quat_waypoints[:, i]
    
    # ============ PASO 6: NORMALIZACIÓN DE VELOCIDADES ============
    for i in range(n_final):
        vel_norm = np.linalg.norm(vel_waypoints[:, i])
        if vel_norm > 1e-6:
            vel_waypoints[:, i] /= vel_norm
    
    # Log info
    if n_final > n_waypoints:
        rospy.loginfo(f"[MPCC] Waypoints refinados: {n_waypoints} -> {n_final} (giros detectados)")
    
    return s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints


def verify_waypoint_interpolation_quality(s_waypoints, pos_waypoints, quat_waypoints,
                                          position_by_arc_length, quaternion_by_arc_length):
    """
    Verifica la calidad de la interpolación comparando waypoints con funciones originales.
    
    Args:
        s_waypoints: Longitudes de arco de los waypoints
        pos_waypoints: Posiciones de los waypoints [3, N]
        quat_waypoints: Quaterniones de los waypoints [4, N]
        position_by_arc_length: Función original de posición
        quaternion_by_arc_length: Función original de quaternion
    
    Returns:
        pos_error_max: Error máximo de posición [m]
        quat_error_max: Error máximo de quaternion (norma)
        quality_ok: True si la interpolación tiene calidad suficiente
    """
    
    pos_error_max = 0.0
    quat_error_max = 0.0
    
    for i in range(len(s_waypoints)):
        # Comparar con función original
        pos_original = position_by_arc_length(s_waypoints[i])
        quat_original = quaternion_by_arc_length(s_waypoints[i])
        
        # Calcular errores
        pos_error = np.linalg.norm(pos_waypoints[:, i] - pos_original)
        quat_error = np.linalg.norm(quat_waypoints[:, i] - quat_original)
        
        pos_error_max = max(pos_error_max, pos_error)
        quat_error_max = max(quat_error_max, quat_error)
    
    # Criterio de calidad: error de posición < 1cm
    quality_ok = pos_error_max < 0.01
    
    return pos_error_max, quat_error_max, quality_ok


def main(odom_pub_1, odom_pub_2, trajec_pub, L, x0, v_max, a_max, n, initial):
    # Split Values
    m = L[0]
    g = L[4]
    # Sample Time Defintion
    t_final = 30
    # Sample time
    frec= 80
    t_s = 1/frec

    # ============ CONFIGURACIÓN DEL EXPERIMENTO MPCC ============
    # El objetivo es completar el recorrido target lo más rápido posible
    # mientras se minimizan los errores de contorno y lag
    TARGET_PATH_LENGTH_M = 80   # Recorrido objetivo en metros
    MIN_PROGRESS_SPEED = 0.2      # [m/s] velocidad mínima para evitar estancamiento
    MAX_PROGRESS_SPEED = 15.0     # [m/s] velocidad máxima - alta para ver si el optimizador la reduce en curvas

    # ============ SELECCIÓN DE TRAYECTORIA ============
    # True: Usar trayectoria de PMM (path planning por gates)
    # False: Usar trayectoria analítica (sinusoidal)
    USE_PMM_TRAJECTORY = True

    sample_time = t_s 
    # Prediction Time
    N_horizont = 10
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # ============ CARGAR TRAYECTORIA ============
    if USE_PMM_TRAJECTORY:
        # Cargar trayectoria PMM (path planning por gates)
        rospy.loginfo("=" * 60)
        rospy.loginfo("[MPCC] Cargando trayectoria PMM...")
        rospy.loginfo("=" * 60)
        
        from pmm_trajectory import load_pmm_trajectory
        pmm = load_pmm_trajectory(verbose=True)
        
        position_by_arc_length = pmm['position_by_arc_length']
        velocity_by_arc_length = pmm['velocity_by_arc_length']
        quaternion_by_arc_length = pmm['quaternion_by_arc_length']
        arc_lengths = pmm['arc_lengths']
        total_path_length = pmm['total_length']
        
        # Actualizar t_final según PMM
        t_final = pmm['total_time']
        t = np.arange(0, t_final + t_s, t_s)
        
        # Actualizar posición inicial del dron
        pos_init = position_by_arc_length(0.0)
        quat_init = quaternion_by_arc_length(0.0)
        x0[0:3] = pos_init
        x0[3:7] = quat_init
        
        rospy.loginfo("[MPCC] Posición inicial actualizada: [%.2f, %.2f, %.2f]", 
                      pos_init[0], pos_init[1], pos_init[2])
        
        # Crear hd y hd_d para compatibilidad (no se usan directamente)
        hd = [pmm['positions'][:, 0], pmm['positions'][:, 1], pmm['positions'][:, 2]]
        hd_d = [pmm['velocities'][:, 0], pmm['velocities'][:, 1], pmm['velocities'][:, 2]]
    else:
        # Trayectoria analítica (sinusoidal)
        rospy.loginfo("=" * 60)
        rospy.loginfo("[MPCC] Usando trayectoria analítica...")
        rospy.loginfo("=" * 60)
        
        xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)
        hd = [xd, yd, zd]
        hd_d = [xd_p, yd_p, zd_p]

        # Calcular posiciones parametrizadas en longitud de arco
        t_finer = np.linspace(0, t_final, len(t))

        arc_lengths, pos_samples, position_by_arc_length, quaternion_by_arc_length, velocity_by_arc_length, spline_t, xd_p, yd_p, zd_p = calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_finer, t_max=t_final)

        total_path_length = arc_lengths[-1]
    
    # ============ TARGET DE RECORRIDO ============
    # Limitar al target configurado o a la longitud total disponible
    target_path_length = min(TARGET_PATH_LENGTH_M, total_path_length)
    
    # Velocidad nominal inicial (se usará como referencia, pero el MPCC optimizará)
    s_nominal_rate = np.clip(target_path_length / t_final, MIN_PROGRESS_SPEED, MAX_PROGRESS_SPEED)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("[MPCC] Configuración del experimento:")
    rospy.loginfo("  - Fuente: %s", "PMM" if USE_PMM_TRAJECTORY else "Analítica")
    rospy.loginfo("  - Longitud total de trayectoria: %.2f m", total_path_length)
    rospy.loginfo("  - Recorrido target: %.2f m", target_path_length)
    rospy.loginfo("  - Velocidad mínima: %.2f m/s", MIN_PROGRESS_SPEED)
    rospy.loginfo("  - Velocidad máxima: %.2f m/s", MAX_PROGRESS_SPEED)
    rospy.loginfo("  - Velocidad nominal inicial: %.2f m/s", s_nominal_rate)
    rospy.loginfo("=" * 60)

    # Perfil nominal de avance sobre la trayectoria (constante en longitud de arco)
    s_profile = np.clip(np.arange(t.shape[0]) * sample_time * s_nominal_rate, 0, total_path_length)

    # Referencias coherentes con la parametrización por longitud de arco
    pos_ref_track = np.zeros((3, t.shape[0]))
    quat_ref_track = np.zeros((4, t.shape[0]))
    vel_ref_track = np.zeros((3, t.shape[0]))

    for idx, s_val in enumerate(s_profile):
        pos_ref_track[:, idx] = position_by_arc_length(s_val)
        quat_ref_track[:, idx] = quaternion_by_arc_length(s_val)
        vel_ref_track[:, idx] = velocity_by_arc_length(s_val)

    # ============ CASADI TRAJECTORY INTERPOLATION FOR MPCC ============
    # Create waypoints for CasADi interpolation using modular function
    rospy.loginfo("[MPCC] Creating waypoints for CasADi interpolation...")
    # PMM tiene giros bruscos - usa más waypoints base y refinamiento adaptivo
    # La función create_mpcc_waypoints añadirá más donde haya giros
    N_WAYPOINTS = 50 if USE_PMM_TRAJECTORY else 30
    
    s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints = create_mpcc_waypoints(
        position_by_arc_length, 
        velocity_by_arc_length,
        quaternion_by_arc_length,
        target_path_length,
        n_waypoints=N_WAYPOINTS
    )
    
    # Verify interpolation quality
    pos_error_max, quat_error_max, quality_ok = verify_waypoint_interpolation_quality(
        s_waypoints, pos_waypoints, quat_waypoints,
        position_by_arc_length, quaternion_by_arc_length
    )
    
    rospy.loginfo("[MPCC] Interpolation quality check:")
    rospy.loginfo("  - Number of waypoints: %d", N_WAYPOINTS)
    rospy.loginfo("  - Average spacing: %.2f m", target_path_length/(N_WAYPOINTS-1))
    rospy.loginfo("  - Max position error: %.6f m", pos_error_max)
    rospy.loginfo("  - Max quaternion error: %.6f", quat_error_max)
    
    if not quality_ok:
        rospy.logwarn("  ⚠ Position error > 1cm! Consider increasing N_WAYPOINTS to 50+")
    else:
        rospy.loginfo("  ✓ Interpolation quality OK")
    
    # Create CasADi interpolation functions
    rospy.loginfo("[MPCC] Creating CasADi trajectory interpolation functions...")
    # Usar siempre gamma_dq para ambos modos (PMM y analítico)
    gamma_dq = create_casadi_dual_quaternion_interpolator(s_waypoints, pos_waypoints, quat_waypoints)
    gamma_pos, gamma_vel, gamma_quat = create_casadi_trajectory_interpolator(
        s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints
    )
    rospy.loginfo("[MPCC] CasADi interpolation functions created successfully")
    rospy.loginfo("[MPCC]   - Arc-length range: [0, %.2f] m", target_path_length)

    vmax = 5
    alpha= 0.2
    #pos_ref, s_progress, v_ref, dp_ds = calculate_reference_positions_and_curvature(arc_lengths, position_by_arc_length, t, t_s, vmax  , alpha)

    # Time defintion aux variable

    # Frequency of the simulation
    hz = int(1/(sample_time))
    # loop_rate usando time.perf_counter (no depende de ROS, más preciso)
    loop_period = sample_time  # [s]

    # ============ HORIZONTE DE PREDICCIÓN ============
    # Aumentado de 0.5s a 0.6s para mejor anticipación de curvas
    # Esto permite que el optimizador "vea más lejos" y frene antes
    # NOTA: No usar horizontes muy largos (>1s) porque aumenta tiempo de cómputo
    t_N = 0.5  # [s] Horizonte de predicción - Balance velocidad/anticipación
    # Prediction Node of the NMPC formulation
    N = np.arange(0, t_N + sample_time, sample_time)
    N_prediction = N.shape[0]

    # Aux variables samplte time
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = sample_time*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Message Ros
    rospy.loginfo_once("DualQuaternion.....")
    message_ros = "DualQuaternion Casadi NMPC "

    # Odometry Message
    quat_1_msg = Odometry()
    quat_1_d_msg = Odometry()
    
    # Defining initial condition of the system and verify properties

    qw1 = x0[1]
    qx1 = x0[4]
    qy1 = x0[5]
    qz1 = x0[6]
    tx1 = x0[0]
    ty1 = x0[1]
    tz1 = x0[2]

    # Initial Dualquaternion
    dual_1 = dualquat_from_pose(qw1, qx1, qy1,  qz1, tx1, ty1, tz1)
    angular_linear_1 = np.array([x0[7], x0[8], x0[9], x0[10], x0[11], x0[12]]) # Angular Body linear Inertial
    dual_twist_1 = dual_twist(angular_linear_1, dual_1)
    velocities  = dual_twist_1

    # Empty Matrices that are used to plot the results
    Q1_trans_data = np.zeros((4, t.shape[0] + 1 - N_prediction), dtype=np.double)
    Q1_quat_data = np.zeros((4, t.shape[0] + 1 - N_prediction), dtype=np.double)
    Q1_velocities_data = np.zeros((6, t.shape[0] + 1 - N_prediction), dtype=np.double)

    # Vector of the generalized states (15 now: dq + w_b + v_b + s)
    X = np.zeros((15, t.shape[0] + 1 - N_prediction), dtype=np.double)
    #X_aux = np.zeros((8, t.shape[0] + 1 - N_prediction), dtype=np.double)

    # Control Actions (5 now: F + tau_1,2,3 + u_s)
    u = np.zeros((5, t.shape[0] - N_prediction), dtype=np.double)
    F = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    M = np.zeros((3, t.shape[0] - N_prediction), dtype=np.double)
    
    # ============ MPCC VELOCITY TRACKING ============
    # Para verificar que u_s (punto virtual) es consistente con v_real (dron)
    u_s_history = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    v_real_history = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    v_tangent_history = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)  # Componente en dirección tangente
    s_history = np.zeros(t.shape[0] - N_prediction, dtype=np.double)  # Arc-length para curvatura

    # Update initial condition in the empty matrices
    Q1_trans_data[:, 0] = np.array(get_trans(dual_1)).reshape((4, ))
    Q1_quat_data[:, 0] = np.array(get_quat(dual_1)).reshape((4, ))
    Q1_velocities_data[:, 0] = np.array(velocities).reshape((6, ))
    
    # Initialize arc length state (NEW): X = [dq (8), w_b (3), v_b (3), s (1)]
    s_initial = 0.0  # Start at beginning of trajectory
    X[:, 0] = np.array(ca.vertcat(dual_1, dual_twist_1, s_initial)).reshape((15, ))
    
    # Initialize arc length state variable (NEW)
    s_current = 0.0  # Start at beginning of trajectory

    # Constraints on control actions
    F_max = L[0]*L[4] + 20
    F_min = 0
    tau_1_max = 0.1
    tau_1_min = -0.1
    tau_2_max = 0.1
    tau_2_min = -0.1
    tau_3_max = 0.1
    taux_3_min = -0.1
    u_s_min = MIN_PROGRESS_SPEED
    u_s_max = MAX_PROGRESS_SPEED

    # Desired Trajectory
    #hd, hd_d, qd, w_d, f_d, M_d = compute_reference(t, sample_time, v_max, a_max, n, L)
    #hd, hd_d, hd_dd, hd_ddd, hd_dddd, theta, theta_d, theta_dd = trajectory(t, 2, (initial +1)*0.5)
    #hd, hd_d, qd, w_d, f_d, M_d = compute_flatness_states(t, L,  2, (initial + 1)*0.5)
    #hd, hd_d, qd, w_d, f_d, M_d = compute_flatness_states(t, L,  2, (initial + 1)*0.5, x0[0:3], sample_time)


    # Initial condition for the desired states (15 now: dq + w_b + v_b + s)
    X_d = np.zeros((15, t.shape[0]+1), dtype=np.double)

    # Desired Reference Inputs (5 now: F + tau_1,2,3 + u_s)
    u_d = np.zeros((5, t.shape[0]), dtype=np.double)

    def set_reference_column(idx, s_value):
        if idx >= X_d.shape[1]:
            return
        pos_val = position_by_arc_length(s_value)
        quat_val = quaternion_by_arc_length(s_value)
        vel_val = velocity_by_arc_length(s_value)
        dq_val = dualquat_from_pose(quat_val[0], quat_val[1], quat_val[2], quat_val[3],
                                     pos_val[0], pos_val[1], pos_val[2])
        X_d[0:8, idx] = np.array(dq_val).reshape((8, ))
        X_d[8:14, idx] = np.array([0.0, 0.0, 0.0, vel_val[0], vel_val[1], vel_val[2]])
        X_d[14, idx] = s_value

    for k in range(0, t.shape[0]):
        u_d[0, k] = 0 #f_d[0, k]
        u_d[1, k] = 0#M_d[0, k]
        u_d[2, k] = 0#M_d[1, k]
        u_d[3, k] = 0#M_d[2, k]
        set_reference_column(k, s_profile[k])


    # Empty vectors for the desired Dualquaernion
    Q2_trans_data = np.zeros((4, t.shape[0] + 1 - N_prediction), dtype=np.double)
    Q2_quat_data = np.zeros((4, t.shape[0] + 1 - N_prediction), dtype=np.double)

    Q2_trans_data[:, 0] = np.array(get_trans(X_d[0:8, 0])).reshape((4, ))
    Q2_quat_data[:, 0] = np.array(get_quat(X_d[0:8, 0])).reshape((4, ))

    # Odometry message
    quat_1_msg = get_odometry(quat_1_msg, X[0:8, 0], 'quat_1')
    send_odometry(quat_1_msg, odom_pub_1)

    quat_1_d_msg = get_odometry(quat_1_d_msg, X_d[0:8, 0], 'quat_1_d')
    send_odometry(quat_1_d_msg, odom_pub_2)

    marker_msg = Marker()
    marker_msg, points = init_marker(marker_msg, pos_ref_track[:, 0])

    # Optimization problem (con límites de velocidad de progreso)
    # NOW passing CasADi interpolation functions for dynamic reference computation
    rospy.loginfo("[MPCC] Creating OCP solver with CasADi trajectory interpolation...")
    ocp = create_ocp_solver(X[:, 0], N_prediction, t_N, F_max, F_min, tau_1_max, tau_1_min, 
                            tau_2_max, tau_2_min, tau_3_max, taux_3_min, L, sample_time,
                            u_s_min=u_s_min, u_s_max=u_s_max,
                            gamma_dq=gamma_dq, gamma_vel=gamma_vel)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= False, generate= False)

    # Integration using Acados
    acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_" + ocp.model.name + ".json", build= True, generate= True)
    #acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_" + ocp.model.name + ".json", build= False, generate= False)

    # Dimensions of the optimization problem
    x_dim = ocp.model.x.size()[0]
    u_dim = ocp.model.u.size()[0]

    # Reset Solver
    acados_ocp_solver.reset()

    # Initial Conditions optimization problem
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", X[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", u_d[:, 0])

    # Noise
    sigma_x = 0.01
    sigma_y = 0.01
    sigma_z = 0.01
    sigma_theta_x = 0.001
    sigma_theta_y = 0.001
    sigma_theta_z = 0.001
    sigma_vx = 0.0001
    sigma_vy = 0.0001
    sigma_vz = 0.0001
    sigma_wx = 0.0001
    sigma_wy = 0.0001
    sigma_wz = 0.0001
    aux_noise = np.zeros(12)
    aux_noise[0] = sigma_x**2
    aux_noise[1] = sigma_y**2
    aux_noise[2] = sigma_z**2
    aux_noise[3] = sigma_theta_x**2
    aux_noise[4] = sigma_theta_y**2
    aux_noise[5] = sigma_theta_z**2
    aux_noise[6] = sigma_vx**2
    aux_noise[7] = sigma_vy**2
    aux_noise[8] = sigma_vz**2
    aux_noise[9] = sigma_wx**2
    aux_noise[10] = sigma_wy**2
    aux_noise[11] = sigma_wz**2
    uav_white_noise_cov = np.diag(aux_noise)

    # Cost Values
    orientation_cost = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    translation_cost = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    control_cost = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    total_cost = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)

    lie_cost = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)

    # KKT conditions
    kkt_values = np.zeros((4, t.shape[0] - N_prediction), dtype=np.double)
    sqp_iteration = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)

    error_dual_no_filter = np.array(error_dual_f(X_d[0:8, 0], X[0:8, 0])).reshape((8, ))
    if error_dual_no_filter[0] > 0.0:
        X_d[0:8, :] = X_d[0:8, :]
    else:
        X_d[0:8, :] = -X_d[0:8, :]
    
    # ============ VARIABLES PARA TRACKING DE PROGRESO ============
    trajectory_completed = False
    completion_time = None
    completion_step = None
    
    # Simulation loop
    for k in range(0, t.shape[0] - N_prediction):
        tic = time.perf_counter()
        
        # ============ VERIFICAR SI SE COMPLETÓ EL RECORRIDO TARGET ============
        if s_current >= target_path_length and not trajectory_completed:
            trajectory_completed = True
            completion_time = k * sample_time
            completion_step = k
            rospy.loginfo("=" * 60)
            rospy.loginfo("[MPCC] ¡RECORRIDO COMPLETADO!")
            rospy.loginfo("  - Recorrido target: %.2f m", target_path_length)
            rospy.loginfo("  - Recorrido actual: %.2f m", s_current)
            rospy.loginfo("  - Tiempo de completación: %.2f s", completion_time)
            rospy.loginfo("  - Velocidad promedio: %.2f m/s", target_path_length / completion_time if completion_time > 0 else 0)
            rospy.loginfo("=" * 60)
            break  # Terminar simulación al completar el target
        
        marker_msg, points = send_marker(marker_msg, points, trajec_pub, pos_ref_track[:, k])
        # Update reference snapshot for current time index based on actual arc progression
        set_reference_column(k, s_current)

        white_noise = 0*np.random.multivariate_normal(np.zeros(12),uav_white_noise_cov)

        # Compute cost
        orientation_cost[:, k] = cost_quaternion(get_quat(X_d[0:8, k]), get_quat(X[0:8, k]))
        translation_cost[:, k] = cost_translation(get_trans(X_d[0:8, k]), get_trans(X[0:8, k]))
        total_cost[:, k] = orientation_cost[:, k] + translation_cost[:, k]

        # Check properties
        real = X[0:4, k]
        dual = X[4:8, k]
        # Chekc norm of the quaternion
        quat_check = get_quat(X[0:8, k])
        #print("-----")
        #print(np.linalg.norm(quat_check))
        #print(np.dot(real, dual))
        # Control Law Acados
        acados_ocp_solver.set(0, "lbx", X[:, k])
        acados_ocp_solver.set(0, "ubx", X[:, k])

        # Desired Trajectory of the system - NOW DYNAMIC BASED ON ARC LENGTH (NEW)
        for j in range(N_prediction):
            # Predict arc length at horizon step j usando la velocidad máxima como referencia
            # El MPCC decidirá la velocidad óptima real
            s_pred_j = s_current + j * sample_time * MAX_PROGRESS_SPEED
            s_pred = np.clip(s_pred_j, 0, target_path_length)  # Limitar al target

            # Get reference at predicted arc length
            pos_ref_j = position_by_arc_length(s_pred)
            quat_ref_j = quaternion_by_arc_length(s_pred)
            vel_ref_j = velocity_by_arc_length(s_pred)
            
            # Build dual quaternion from references
            dq_ref_j = dualquat_from_pose(quat_ref_j[0], quat_ref_j[1], quat_ref_j[2], quat_ref_j[3],
                                          pos_ref_j[0], pos_ref_j[1], pos_ref_j[2])
            
            # Reference velocities (angular = 0, linear = velocity direction)
            vel_ref_nominal = np.array([0.0, 0.0, 0.0, vel_ref_j[0], vel_ref_j[1], vel_ref_j[2]])
            
            # Build state reference X_d[:, k+j]
            yref = np.hstack([np.array(dq_ref_j).reshape(8), vel_ref_nominal])
            # NOTE: u_d now has 5 elements (F, tau1, tau2, tau3, u_s)
            # But parameters only need first 4 (without u_s, which is optimized by MPC)
            uref = u_d[0:4, k+j]  # Only first 4 controls as reference
            aux_ref = np.hstack((yref, uref))
            acados_ocp_solver.set(j, "p", aux_ref)
        # Desired Trajectory at the last Horizon (also dynamic now)
        s_pred_Nj = s_current + N_prediction * sample_time * MAX_PROGRESS_SPEED
        s_pred_N = np.clip(s_pred_Nj, 0, target_path_length)
        pos_ref_N = position_by_arc_length(s_pred_N)
        quat_ref_N = quaternion_by_arc_length(s_pred_N)
        vel_ref_N = velocity_by_arc_length(s_pred_N)
        dq_ref_N = dualquat_from_pose(quat_ref_N[0], quat_ref_N[1], quat_ref_N[2], quat_ref_N[3],
                                      pos_ref_N[0], pos_ref_N[1], pos_ref_N[2])
        vel_ref_nominal_N = np.array([0.0, 0.0, 0.0, vel_ref_N[0], vel_ref_N[1], vel_ref_N[2]])
        yref_N = np.hstack([np.array(dq_ref_N).reshape(8), vel_ref_nominal_N])
        # Only first 4 controls as reference
        uref_N = u_d[0:4, k+N_prediction]
        aux_ref_N = np.hstack((yref_N, uref_N))
        acados_ocp_solver.set(N_prediction, "p", aux_ref_N)

        # Check Solution since there can be possible errors 
        #acados_ocp_solver.options_set("rti_phase", 2)
        acados_ocp_solver.solve()

        stat_fields = ['statistics', 'time_tot', 'time_lin', 'time_sim', 'time_sim_ad', 'time_sim_la', 'time_qp', 'time_qp_solver_call', 'time_reg', 'sqp_iter', 'residuals', 'qp_iter', 'alpha']

        for field in stat_fields:
            #print(f"{field} : {acados_ocp_solver.get_stats(field)}")
            None
        print(initial)
        kkt_values[:, k]  = acados_ocp_solver.get_stats('residuals')
        sqp_iteration[:, k] = acados_ocp_solver.get_stats('sqp_iter')
        #acados_ocp_solver.print_statistics()

        # Get the control Action (now includes u_s)
        aux_control = acados_ocp_solver.get(0, "u")
        F[:, k] = aux_control[0]
        M[:, k] = aux_control[1:4]
        u_s_optimal = aux_control[4]  # Arc length rate control (NEW)
        
        u[0, k] = F[:, k]
        u[1:4, k] = M[:, k]
        u[4, k] = u_s_optimal
        
        # ============ MPCC VELOCITY TRACKING ============
        # Guardar u_s (velocidad punto virtual) y arc-length actual
        u_s_history[:, k] = u_s_optimal
        s_history[k] = s_current
        
        # Calcular velocidad real del dron en frame INERCIAL
        v_body = X[11:14, k]  # v_b del estado actual (frame cuerpo)
        q_real = X[0:4, k]    # cuaternión real [w, x, y, z]
        
        # Rotar v_body al frame inercial: v_inertial = R @ v_body
        # Usando la fórmula del cuaternión: v' = q ⊗ v ⊗ q*
        def quat_rotate(q, v):
            """Rota vector v por cuaternión q = [w, x, y, z]"""
            w, x, y, z = q[0], q[1], q[2], q[3]
            # Matriz de rotación del cuaternión
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ])
            return R @ v
        
        v_inertial = quat_rotate(q_real, v_body)
        v_real_magnitude = np.linalg.norm(v_inertial)
        v_real_history[:, k] = v_real_magnitude
        
        # Velocidad en dirección tangente (proyección sobre tangente de trayectoria)
        # Ahora ambos están en frame inercial
        tangent_dir = velocity_by_arc_length(s_current)
        tangent_unit = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-8)
        v_tangent = np.dot(v_inertial, tangent_unit)  # Componente en dirección de avance
        v_tangent_history[:, k] = np.abs(v_tangent)
        
        # Log cada 50 iteraciones para monitorear
        if k % 50 == 0:
            print(f"[k={k:4d}] u_s*={u_s_optimal:6.2f} m/s | v_real={v_real_magnitude:6.2f} m/s | v_tang={v_tangent:6.2f} m/s | ratio={v_tangent/(u_s_optimal+1e-8):5.2f}")

        control_cost[:, k] = np.dot(u[:, k], u[:, k])

        # Timing: esperar el tiempo restante del período sin usar ROS
        toc_solver = time.perf_counter() - tic
        remaining = loop_period - toc_solver
        if remaining > 0:
            time.sleep(remaining)
        delta_t[:, k] = time.perf_counter() - tic

        # Update Data of the system (only first 4 controls for dynamics)
        acados_integrator.set("x", X[:, k])
        acados_integrator.set("u", u[:, k])

        status_integral = acados_integrator.solve()
        xcurrent = acados_integrator.get("x")

        # Update arc length state (NEW)
        # El MPCC optimiza u_s_optimal para maximizar progreso
        # Solo aplicamos los límites físicos, sin forzar un mínimo artificial
        progress_rate = u_s_optimal if np.isfinite(u_s_optimal) else s_nominal_rate
        progress_rate = np.clip(progress_rate, u_s_min, u_s_max)
        s_current += progress_rate * sample_time
        s_current = np.clip(s_current, 0, target_path_length)  # Limitar al target

        # Update Data of the system
        # NOTE: xcurrent from integrator has 14 elements (no arc state)
        # We add arc state manually after integration
        X[:, k+1] = noise(xcurrent, white_noise)
        # CRITICAL: Update arc length state explicitly
        X[14, k+1] = s_current

        # Prepare desired state for next time step using the updated arc progression
        set_reference_column(k+1, s_current)

        # Update Matrices of our system
        Q1_trans_data[:, k + 1] = np.array(get_trans(X[0:8, k+1])).reshape((4, ))
        Q1_quat_data[:, k + 1] = np.array(get_quat(X[0:8, k+1])).reshape((4, ))
        # Compute body angular velocity and inertial linear velocity
        velocities  = X[8:14, k +1]
        # Save body angular and inertial linear velocitu
        Q1_velocities_data[:, k + 1] = np.array(velocities).reshape((6, ))

        # Update matrices of the reference
        Q2_trans_data[:, k+1] = np.array(get_trans(X_d[0:8, k+1])).reshape((4, ))
        Q2_quat_data[:, k+1] = np.array(get_quat(X_d[0:8, k+1])).reshape((4, ))

        # Send Data throught Ros
        quat_1_msg = get_odometry(quat_1_msg, X[0:8, k+1], 'quat_1')
        send_odometry(quat_1_msg, odom_pub_1)

        quat_1_d_msg = get_odometry(quat_1_d_msg, X_d[0:8, k+1], 'quat_1_d')
        send_odometry(quat_1_d_msg, odom_pub_2)

        rospy.loginfo(message_ros + str(total_cost[:, k]))
    
    # Normalize cost
    orientation_cost = orientation_cost/1
    translation_cost = translation_cost/1
    control_cost = control_cost/1
    total_cost = total_cost/np.max(total_cost)

    # ============================================================
    # ÍNDICE DE PARTE ACTIVA (solo hasta donde el dron realmente llegó)
    # completion_step es el paso k exacto donde se completó el recorrido
    # Q1_trans_data se llena en k+1, así que el último dato real es completion_step
    # ============================================================
    t_plot = t[0:u_s_history.shape[1]]
    active_mask = u_s_history.flatten() > 0.1  # siempre definida para estadísticas
    if completion_step is not None:
        # Usar el paso de completación exacto: datos reales hasta completion_step (inclusive)
        last_active_idx = completion_step
    else:
        # Fallback: buscar por máscara de actividad si no hubo completación
        if np.any(active_mask):
            last_active_idx = np.where(active_mask)[0][-1] + 1
        else:
            last_active_idx = len(t_plot)
    last_active_idx = min(last_active_idx, len(t_plot))
    t_active = t_plot[0:last_active_idx]

    # ============================================================
    # PLOTTING SECTION: All States, Control Actions, and Costs
    # Solo se grafica la parte activa del experimento
    # ============================================================
    
    # 1. QUATERNION STATES (actual vs desired)
    fig11, ax11, ax12, ax13, ax14 = fancy_plots_4()
    plot_states_quaternion(fig11, ax11, ax12, ax13, ax14, Q1_quat_data[0:4, :last_active_idx+1], Q2_quat_data[0:4, :last_active_idx+1], t_active, "1_Quaternion_States", folder_path)

    # 2. POSITION STATES (actual vs desired)
    fig21, ax21, ax22, ax23 = fancy_plots_3()
    plot_states_position(fig21, ax21, ax22, ax23, Q1_trans_data[1:4, :last_active_idx+1], Q2_trans_data[1:4, :last_active_idx+1], t_active, "2_Position_States", folder_path)
    
    # 3. ANGULAR VELOCITIES (ωx, ωy, ωz)
    fig31, ax31, ax32, ax33 = fancy_plots_3()
    plot_angular_velocities(fig31, ax31, ax32, ax33, Q1_velocities_data[0:3, :last_active_idx+1], t_active, "3_Angular_Velocities", folder_path)
    
    # 4. LINEAR VELOCITIES (vx, vy, vz)
    fig41, ax41, ax42, ax43 = fancy_plots_3()
    plot_linear_velocities(fig41, ax41, ax42, ax43, Q1_velocities_data[3:6, :last_active_idx+1], t_active, "4_Linear_Velocities", folder_path)
    
    # 5. CONTROL ACTIONS (Thrust and Torques)
    fig51, ax51, ax52, ax53, ax54 = fancy_plots_4()
    plot_control_actions(fig51, ax51, ax52, ax53, ax54, F[:, :last_active_idx], M[:, :last_active_idx], t_active, "5_Control_Actions", folder_path)

    # 6. COST COMPONENTS ANALYSIS
    # 6a. Total Cost
    fig101, ax101 = fancy_plots_1()
    plot_cost_total(fig101, ax101, total_cost[:, :last_active_idx], t_active, "6_Total_Cost", folder_path)
    
    # 6b. Orientation Cost (Quaternion Error)
    fig102, ax102 = fancy_plots_1()
    plot_cost_orientation(fig102, ax102, orientation_cost[:, :last_active_idx], t_active, "7_Orientation_Cost", folder_path)
    
    # 6c. Translation Cost (Position Error)
    fig103, ax103 = fancy_plots_1()
    plot_cost_translation(fig103, ax103, translation_cost[:, :last_active_idx], t_active, "8_Translation_Cost", folder_path)
    
    # 6d. Control Cost (Input Effort)
    fig104, ax104 = fancy_plots_1()
    plot_cost_control(fig104, ax104, control_cost[:, :last_active_idx], t_active, "9_Control_Cost", folder_path)

    # 7. COMPUTATIONAL TIME
    fig105, ax105 = fancy_plots_1()
    plot_time(fig105, ax105, delta_t[:, :last_active_idx], t_sample[:, :last_active_idx], t_active, "10_Computational_Time", folder_path)
    
    # ============ MPCC VELOCITY COMPARISON + CURVATURE PLOT ============
    u_s_active    = u_s_history.flatten()[0:last_active_idx]
    v_real_active = v_real_history.flatten()[0:last_active_idx]
    v_tang_active = v_tangent_history.flatten()[0:last_active_idx]
    s_active      = s_history[0:last_active_idx]
    plot_curvature_vs_velocity(t_active, s_active, u_s_active, v_tang_active,
                               position_by_arc_length,
                               "11_MPCC_Velocity_Comparison", folder_path)

    # ============ FIGURA 12: TRAYECTORIA 3D + VISTAS ORTOGONALES ============
    Q1_active = Q1_trans_data[:, :last_active_idx + 1]
    Q2_active = Q2_trans_data[:, :last_active_idx + 1]
    plot_trajectory_3d(Q1_active, Q2_active, "12a_Trajectory_3D_Isometric", folder_path)
    plot_trajectory_xy(Q1_active, Q2_active, "12b_Trajectory_XY_Plane",     folder_path)
    plot_trajectory_xz(Q1_active, Q2_active, "12c_Trajectory_XZ_Plane",     folder_path)
    
    # Estadísticas finales (solo del período activo)
    u_s_active_stats = u_s_history.flatten()[active_mask]
    v_real_active_stats = v_real_history.flatten()[active_mask]
    v_tang_active_stats = v_tangent_history.flatten()[active_mask]
    
    print("\n" + "="*60)
    print("MPCC VELOCITY ANALYSIS (solo período activo)")
    print("="*60)
    print(f"u_s* promedio:     {np.mean(u_s_active_stats):6.2f} m/s")
    print(f"u_s* máximo:       {np.max(u_s_active_stats):6.2f} m/s")
    print(f"v_real promedio:   {np.mean(v_real_active_stats):6.2f} m/s")
    print(f"v_tangent promedio:{np.mean(v_tang_active_stats):6.2f} m/s")
    print(f"Ratio v_tang/u_s:  {np.mean(v_tang_active_stats)/(np.mean(u_s_active_stats)+1e-8):6.2f}")
    print("="*60)
    print("Si ratio ≈ 1.0 → El dron sigue bien al punto virtual")
    print("Si ratio < 1.0 → El punto virtual va más rápido que el dron")
    print("="*60 + "\n")

    # Timing stats
    dt_active = delta_t.flatten()[:last_active_idx] * 1000  # ms
    print("="*60)
    print("TIMING ANALYSIS (solo período activo)")
    print("="*60)
    print(f"  dt_actual medio:   {np.mean(dt_active):6.2f} ms  (target={sample_time*1000:.1f} ms)")
    print(f"  dt_actual máximo:  {np.max(dt_active):6.2f} ms")
    print(f"  dt_actual P95:     {np.percentile(dt_active, 95):6.2f} ms")
    print(f"  Frecuencia media:  {1000/np.mean(dt_active):6.1f} Hz  (target={1/sample_time:.0f} Hz)")
    print("="*60 + "\n")

    return X, X_d, u, t, F, M, orientation_cost, translation_cost, control_cost, N_prediction, kkt_values, sqp_iteration

if __name__ == '__main__':
    try:
        # Initialization Node
        rospy.init_node("DualQuaternions", disable_signals=True, anonymous=True)
        odomety_topic_1 = "/" + "dual_1" + "/odom"
        odometry_publisher_1 = rospy.Publisher(odomety_topic_1, Odometry, queue_size = 10)

        odomety_topic_2 = "/" + "dual_2" + "/odom"
        odometry_publisher_2 = rospy.Publisher(odomety_topic_2, Odometry, queue_size = 10)

        trajectory_topic = "/" + "ref"
        trajectory_publisher = rospy.Publisher(trajectory_topic, Marker, queue_size = 10)
        # Dynamics Parameters
        m = 1                                                                             
        Jxx = 2.64e-3
        Jyy = 2.64e-3
        Jzz = 4.96e-3
        g = 9.8
        L = [m, Jxx, Jyy, Jzz, g]

        # empty matrices
        Data_States = []
        Data_reference = []
        Data_u = []
        Data_F = []
        Data_M = []
        Data_orientation_cost = []
        Data_translation_cost = []
        Data_control_cost = []
        Data_time = []
        Data_N_prediction = []
        Data_KKT = []
        Data_sqp = []

        print(x_0.shape[0])

        #a_max = np.array([1 ,2, 3, 4])*0.3
        #v_max = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4])*1

        a_max = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])*0.3
        v_max = np.array([1])*1

        # Use itertools.product to get all possible combinations
        combinations = np.array(list(itertools.product(v_max, a_max)))
        print(combinations)
        number_experiments = combinations.shape[0]
        print(number_experiments)

        # Multiple Experiments
        #for k in range(x_0.shape[0]):
        k=1
        # Extend x_0 from 13 to 15 elements: add arc length state s=0 at the end
        x0_extended = np.concatenate([x_0[k, :], [0.0]])
        x, xref, u_data, t, F, M, orientation_cost, translation_cost, control_cost, N_prediction, kkt_values, sqp_iteration = main(odometry_publisher_1, odometry_publisher_2, trajectory_publisher, L, x0_extended,  combinations[k, 0], combinations[k, 1], 1, k)
        Data_States.append(x)
        Data_reference.append(xref)
        Data_u.append(u_data)
        Data_F.append(F)
        Data_M.append(M)
        Data_orientation_cost.append(orientation_cost)
        Data_translation_cost.append(translation_cost)
        Data_control_cost.append(control_cost)
        Data_time.append(t)
        Data_N_prediction.append(N_prediction)
        Data_KKT.append(kkt_values)
        Data_sqp.append(sqp_iteration)

        Data_States = np.array(Data_States)
        Data_reference = np.array(Data_reference)
        Data_u = np.array(Data_u)
        Data_F = np.array(Data_F)
        Data_M = np.array(Data_M)
        Data_orientation_cost = np.array(Data_orientation_cost)
        Data_translation_cost = np.array(Data_translation_cost)
        Data_control_cost = np.array(Data_control_cost)
        Data_time = np.array(Data_time)
        Data_N_prediction = np.array(Data_N_prediction)
        Data_KKT = np.array(Data_KKT)
        Data_sqp = np.array(Data_sqp)

        # Save Data matlab 
        mdic_x = {"X": Data_States, "X_d": Data_reference, "u": Data_u, "F": Data_F, "M": Data_M, "orientation_cost": Data_orientation_cost,
                "translation_cost": Data_translation_cost, "control_cost": Data_control_cost, "t": Data_time, "N": Data_N_prediction, 'KKT': Data_KKT, 'sqp': Data_sqp}
        savemat("Dual_cost_identification.mat", mdic_x)
        #savemat("Dual_cost_without_velocities" + ".mat", mdic_x)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        pass
    else:
        print("Complete Execution")
        pass