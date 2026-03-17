from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat, horzcat, vertsplit
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import norm_2
from casadi import cross
from casadi import if_else
from casadi import atan2
from casadi import exp

from casadi import jacobian
from casadi import sqrt
from casadi import substitute
from scipy.spatial.transform import Rotation as R
import math
import scipy.io
import time as time_module

# CARGA FUNCIONES DEL PROGRAMA
from graficas import plot_pose, plot_error, plot_time, plot_control, plot_vel_lineal, plot_vel_angular, plot_CBF

#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

# Global variables Odometry Drone Condicion inicial
x_real = 3
y_real = 3
z_real = 2
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0

# Angular velocities
qx_real = 0
qy_real = 0.0
qz_real = 0
qw_real = 1
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0

hdp_vision = [0,0,0,0,0,0.0]
axes = [0,0,0,0,0,0]


def Rot_zyx(x):
    phi = x[3, 0]
    theta = x[4, 0]
    psi = x[5, 0]

    # Rot Matrix axis X
    RotX = MX.zeros(3, 3)
    RotX[0, 0] = 1.0
    RotX[1, 1] = cos(phi)
    RotX[1, 2] = -sin(phi)
    RotX[2, 1] = sin(phi)
    RotX[2, 2] = cos(phi)

    # Rot Matrix axis Y
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

    R = RotZ@RotY@RotX
    return R



def QuatToRot(quat):
    # Quaternion to Rotational Matrix
    q = quat # Convierte la lista de cuaterniones en un objeto MX
    
    # Calcula la norma 2 del cuaternión
    q_norm = norm_2(q)
    
    # Normaliza el cuaternión dividiendo por su norma
    q_normalized = q / q_norm

    q_hat = MX.zeros(3, 3)

    q_hat[0, 1] = -q_normalized[3]
    q_hat[0, 2] = q_normalized[2]
    q_hat[1, 2] = -q_normalized[1]
    q_hat[1, 0] = q_normalized[3]
    q_hat[2, 0] = -q_normalized[2]
    q_hat[2, 1] = q_normalized[1]

    Rot = MX.eye(3) + 2 * q_hat @ q_hat + 2 * q_normalized[0] * q_hat

    return Rot

def quaternion_multiply(q1, q2):
    # Descomponer los cuaterniones en componentes
    w0, x0, y0, z0 = vertsplit(q1)
    w1, x1, y1, z1 = vertsplit(q2)
    
    # Calcular la parte escalar
    scalar_part = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    
    # Calcular la parte vectorial
    vector_part = vertcat(
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    )
    
    # Combinar la parte escalar y vectorial
    q_result = vertcat(scalar_part, vector_part)
    
    return q_result


def quat_p(quat, omega):
    # Crear un cuaternión de omega con un componente escalar 0
    omega_quat = vertcat(MX(0), omega)
    
    # Calcular la derivada del cuaternión
    q_dot = 0.5 * quaternion_multiply(quat, omega_quat)
    
    return q_dot

def quaternion_error(q_real, quat_d):
    norm_q = norm_2(q_real)
   
    
    q_inv = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    
    q_error = quaternion_multiply(q_inv, quat_d)
    return q_error

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode_complete'
    # Dynamic Values of the system
    m = 1
    e = MX([0, 0, 1])
    g = 9.81
    


    # set up states & controls
    # Position
    p1 = MX.sym('p1')
    p2 = MX.sym('p2')
    p3 = MX.sym('p3')
    # Orientation
    v1 = MX.sym('v1')
    v2 = MX.sym('v2')
    v3 = MX.sym('v3')

    # Velocity Linear and Angular
    q0 = MX.sym('q0')
    q1 = MX.sym('q1')
    q2 = MX.sym('q2')
    q3 = MX.sym('q3')

    # Velocidades angulares
    w1 = MX.sym('w1')
    w2 = MX.sym('w2')
    w3 = MX.sym('w3')

    # General vector of the states
    x = vertcat(p1, p2, p3, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3)

    # Action variables
    Tt = MX.sym('Tt')
    tau1 = MX.sym('tau1')
    tau2 = MX.sym('tau2')
    tau3 = MX.sym('tau3')

    # General Vector Action variables
    u = vertcat(Tt, tau1, tau2, tau3)

    # Variables to explicit function
 
    p1_p = MX.sym('p1_p')
    p2_p = MX.sym('p2_p')
    p3_p = MX.sym('p3_p')

    v1_p = MX.sym('v1_p')
    v2_p = MX.sym('v2_p')
    v3_p = MX.sym('v3_p')

    q0_p = MX.sym('q0')
    q1_p = MX.sym('q1')
    q2_p = MX.sym('q2')
    q3_p = MX.sym('q3')

    w1_p = MX.sym('w1_p')
    w2_p = MX.sym('w2_p')
    w3_p = MX.sym('w3_p')

    # general vector X dot for implicit function
    x_p = vertcat(p1_p, p2_p, p3_p, v1_p, v2_p, v3_p, q0_p, q1_p, q2_p, q3_p, w1_p, w2_p, w3_p)

    # Ref system as a external value
    p1_d = MX.sym('p1_d')
    p2_d = MX.sym('p2_d')
    p3_d = MX.sym('p3_d')
    
    v1_d = MX.sym('v1_d')
    v2_d = MX.sym('v2_d')
    v3_d = MX.sym('v3_d')

    q0_d = MX.sym('q0_d')
    q1_d = MX.sym('q1_d')
    q2_d = MX.sym('q2_d')
    q3_d = MX.sym('q3_d')

    w1_d = MX.sym('w1_d')
    w2_d = MX.sym('w2_d')
    w3_d = MX.sym('w3_d')

    T_d = MX.sym('T_d')
    tau1_d = MX.sym('tau1_d')
    tau2_d = MX.sym('tau2_d')
    tau3_d = MX.sym('tau3_d')
    
    p = vertcat(p1_d, p2_d, p3_d, v1_d, v2_d, v3_d, q0_d, q1_d, q2_d, q3_d, w1_d, w2_d, w3_d, T_d, tau1_d, tau2_d, tau3_d)

    # Crea una lista de MX con los componentes del cuaternión
    quat = vertcat(q0, q1, q2, q3)
    w = vertcat(w1, w2, w3)
    Rot = QuatToRot(quat)
    # Definición de la matriz de inercia I
    
    Jxx = 0.00305587
    Jyy = 0.00159695
    Jzz = 0.00159687

    I = vertcat(
        horzcat(Jxx, 0, 0),
        horzcat(0, Jyy, 0),
        horzcat(0, 0, Jzz)
    )

    u1 = vertcat(0, 0, Tt)
    u2 = vertcat(tau1, tau2, tau3)

    p_p = vertcat(v1, v2, v3)
    v_p = -e*g + ((Rot @ u1)  / m) 

    

    q_p = quat_p(quat, w)  

    

    w_p = inv(I) @ (u2 - cross(w, I @ w))

    f_expl = vertcat(
        p_p,
        v_p,
        q_p,
        w_p
    )

    # Define f_x and g_x
    # Parte libre: evalúa f(x,u) en u=0
    u_zero = MX.zeros(u.shape[0], 1)
    f0_expr = substitute(f_expl, u, u_zero)   # f0(x) - dinámica libre sin control
    
    # Crear funciones CasADi
    f_x = Function('f0', [x], [f0_expr])                          # dinámica libre
    g_x = Function('g', [x], [jacobian(f_expl, u)])              # incidencia de u

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model

    f_impl = x_p - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_p
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system, f_x, g_x


def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    aux_x = np.array(x[:,0]).reshape((13,))
    return aux_x


def f_yaw():
    value = axes[2]
    return value

def RK4_yaw(x, ts, f_yaw):
    k1 = f_yaw()
    k2 = f_yaw()
    k3 = f_yaw()
    k4 = f_yaw()
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4) 
    aux_x = np.array(x[0]).reshape((1,))
    return aux_x

def Angulo(ErrAng):
    
    if ErrAng >= math.pi:
        while ErrAng >= math.pi:
            ErrAng = ErrAng - 2 * math.pi
        return ErrAng

    # Limitar el ángulo entre [-pi : 0]
    if ErrAng <= -math.pi:
        while ErrAng <= -math.pi:
            ErrAng = ErrAng + 2 * math.pi
        return ErrAng

    return ErrAng



def visual_callback(msg):

    global hdp_vision 

    vx_visual = msg.twist.linear.x
    vy_visual = msg.twist.linear.y
    vz_visual = msg.twist.linear.z
    wx_visual = msg.twist.angular.x
    wy_visual = msg.twist.angular.y
    wz_visual = msg.twist.angular.z

    hdp_vision = [vx_visual, vy_visual, vz_visual, wx_visual, wy_visual, wz_visual]
    

def log_cuaternion_casadi(q):
 

    # Descomponer el cuaternio en su parte escalar y vectorial
    q_w = q[0]
    q_v = q[1:]

    q = if_else(
        q_w < 0,
        -q,  # Si q_w es negativo, sustituir q por -q
        q    # Si q_w es positivo o cero, dejar q sin cambios
    )

    # Actualizar q_w y q_v después de cambiar q si es necesario
    q_w = q[0]
    q_v = q[1:]
    
    # Calcular la norma de la parte vectorial usando CasADi
    norm_q_v = norm_2(q_v)

    #print(norm_q_v)
    
    # Calcular el ángulo theta
    theta = atan2(norm_q_v, q_w)
    
    log_q = 2 * q_v * theta / norm_q_v
    
    return log_q

def create_ocp_solver_description(x0, N_horizon, t_horizon) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system, f_x, g_x = f_system_model()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    np_param = model.p.size()[0]

    # set dimensions
    ocp.solver_options.N_horizon = N_horizon

    # Matriz de ganancia Posicion
    Q_mat = MX.zeros(3, 3)
    Q_mat[0, 0] = 25   # Penalización del error de posición XY
    Q_mat[1, 1] = 25
    Q_mat[2, 2] = 30   # Penalización en Z (altura)

    K_mat = MX.zeros(3, 3)
    K_mat[0, 0] = 12   # Penalización del error de orientación
    K_mat[1, 1] = 12
    K_mat[2, 2] = 12
    
    # Matriz de ganancia Acciones de contol - MÁS PENALIZACIÓN PARA SUAVIDAD
    R_mat = MX.zeros(4, 4)
    R_mat[0, 0] = 1.0    # AUMENTADO: Penalización del thrust para suavidad
    R_mat[1, 1] = 800    # AUMENTADO MUCHO: Penalización de torques para evitar oscilaciones
    R_mat[2, 2] = 800
    R_mat[3, 3] = 800

    ocp.parameter_values = np.zeros(np_param)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = model.p[0:3] - model.x[0:3]
    quat_error = quaternion_error(model.x[6:10], model.p[6:10])

    log_q = log_cuaternion_casadi(quat_error)

    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u + log_q.T @ K_mat @ log_q
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose +  log_q.T @  K_mat @ log_q

   
    obs_x = 2.9     # Posicion donde actualiza la posicion del obstaculo
    obs_y = 0.17
    obs_r = 0.4
    rob_r = 0.35
    
    # PRIMERA BARRERA
    h = sqrt((model.x[0] - obs_x)**2 + (model.x[1] - obs_y)**2) - (rob_r + obs_r)  

    # SEGUNDA BARRERA 
    #distance = sqrt((model.x[0] - obs_x)**2 + (model.x[1] - obs_y)**2)
    
    #h = (distance - (rob_r + obs_r))**2

    f_x_val = f_x(model.x)
    g_x_val = g_x(model.x)


    # Derivada de Lie de primer orden
    Lf_h = jacobian(h, model.x) @ f_x_val 
    Lg_h = jacobian(h, model.x) @ g_x_val

    # Derivada de Lie de segundo orden
    Lf2_h = jacobian(Lf_h, model.x) @ f_x_val
    Lg_L_f_h = jacobian(Lf_h, model.x) @ g_x_val 

    # Derivada de Lie de tercer orden
    Lf3_h = jacobian(Lf2_h, model.x) @ f_x_val 
    Lg_L_f2_h = jacobian(Lf2_h, model.x) @ g_x_val  

    # Barreras temporales
    h_p = Lf_h + Lg_h @ model.u
    h_pp = Lf2_h + Lg_L_f_h @ model.u
    h_ppp = Lf3_h + Lg_L_f2_h @ model.u
  
    # set constraints
    ocp.constraints.constr_type = 'BGH'

    nb_1 = h
    nb_2 = vertcat(h, Lf_h) 

    K_alpha = MX([15, 10]).T  # Parámetros de amortiguamiento de barrera relativa
 

 
   # Define las restricciones 
    constraints = vertcat(h_p + 5*h, h_pp +  K_alpha @ nb_2 )  # Coeficiente CBF
   # constraints = vertcat( h_pp +  K_alpha @ nb_2 )

    # constraints = vertcat(B_p + 1*B)
    # constraints = vertcat(model.x[1])

    # Asigna las restricciones al modelo del OCP
    ocp.model.con_h_expr = constraints
    Dim_constraints = 2  # Define el número total de restricciones

    # Asigna los límites inferiores para las restricciones
    ocp.constraints.lh = np.array([0,0])  # Límite inferior 
    # No se usa el límite inferior de las restricciones en el estado final (opcional)
    # ocp.constraints.lh_e = -1e9 * np.ones(Dim_constraints)

    # Asigna los límites superiores para las restricciones
    ocp.constraints.uh = np.array([1e9,1e9])  # Límite superior
    # No se usa el límite superior de las restricciones en el estado final (opcional)
    # ocp.constraints.uh_e = np.zeros(Dim_constraints)

    # Configuración de las restricciones suaves
    cost_weights = np.ones(Dim_constraints)  # Pesos para las restricciones suaves
    ocp.cost.zu = cost_weights  # Pesos para la penalización en el término de costo de las restricciones suaves (superior)
    ocp.cost.zl = cost_weights  # Pesos para la penalización en el término de costo de las restricciones suaves (inferior)
    ocp.cost.Zl = 10 * cost_weights  # Escala para el costo de las restricciones suaves (inferior)
    ocp.cost.Zu = 10 * cost_weights  # Escala para el costo de las restricciones suaves (superior)

    # Índices para las restricciones suaves (necesario si se usan)
    ocp.constraints.idxsh = np.arange(Dim_constraints)  # Índices de las restricciones suaves

  

    # set constraints
    Tmax = 3*9.81
    taux_max = 0.05
    tauy_max = 0.05
    tauz_max = 0.05

    Tmin = 0 
    taux_min = -taux_max
    tauy_min = - tauy_max
    tauz_min = -tauz_max

    ocp.constraints.lbu = np.array([Tmin,taux_min,tauy_min,tauz_min])
    ocp.constraints.ubu = np.array([Tmax,taux_max,tauy_max,tauz_max])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = x0

    # Restricciones de z

    zmin=1.5
    zmax=50
    #ocp.constraints.lbx = np.array([zmin])
    #ocp.constraints.ubx = np.array([zmax])
    #ocp.constraints.idxbx = np.array([2])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def Euler_p(omega, euler):
    W = np.array([[1, np.sin(euler[0])*np.tan(euler[1]), np.cos(euler[0])*np.tan(euler[1])],
                  [0, np.cos(euler[0]), np.sin(euler[0])],
                  [0, np.sin(euler[0])/np.cos(euler[1]), np.cos(euler[0])/np.cos(euler[1])]])

    euler_p = np.dot(W, omega)
    return euler_p




def send_velocity_control(u, vel_pub=None, vel_msg=None):
    # Local function - no ROS publishing
    # Split control values
    F = u[0]
    tx = u[1]
    ty = u[2]
    tz = u[3]
    
    # Local processing only (no ROS)
    # If ROS publishers are provided, they would be used here
    # But in local mode, this is just a placeholder
    return None


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

def send_full_state_to_sim(state_vector):
    # Local simulation function - no ROS publishing
    # Store state for local use if needed
    pass


def rc_callback(data):
    # Local callback - not used without ROS
    pass


def get_odometry_complete():

    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real

    quaternion = [qx_real, qy_real, qz_real, qw_real ]
    r_quat = R.from_quat(quaternion)
    q2e =  r_quat.as_euler('zyx', degrees = False)
    phi = q2e[2]
    theta = q2e[1]
    psi = q2e[0]

    omega = [wx_real, wy_real, wz_real]
    euler = [phi, theta, psi]
    euler_p = Euler_p(omega,euler)

    x_state = [x_real,y_real,z_real,vx_real,vy_real,vz_real, qw_real, qx_real, qy_real, qz_real, wx_real, wy_real, wz_real ]

    return x_state




    
    
def print_state_vector(state_vector):

# Encabezados de los estados
    headers = ["px", "py", "pz", "vx", "vy", "vz", "qx", "qx", "qy", "qz", "w_x", "w_y", "w_z"]
    
    # Verificar que el tamaño del vector de estado coincida con la cantidad de encabezados
    if len(state_vector) != len(headers):
        raise ValueError(f"El vector de estado tiene {len(state_vector)} elementos, pero se esperaban {len(headers)} encabezados.")
    
    # Determinar la longitud máxima de los encabezados para formateo
    max_header_length = max(len(header) for header in headers)
    
    # Imprimir cada encabezado con el valor correspondiente
    for header, value in zip(headers, state_vector):
        formatted_header = header.ljust(max_header_length)
        print(f"{formatted_header}: {value:.2f}")
    
    # Imprimir una línea en blanco para separación
    print()

def publish_matrix(matrix_data, topic_name='/nombre_del_topico'):
    # Local function - no ROS publishing
    pass

def main():
    # Initial Values System
    # Simulation Time
    t_final = 30
    # Sample time
    frec= 100
    t_s = 1/frec
    # Prediction Horizon
    t_prediction = 0.5           # Tiempo de predicción en segundos (modificar aquí)
    N_prediction = 30            # Número de pasos del MPC (fijo)

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    CBF_value = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)


    # Vector Initial conditions
    x = np.zeros((13, t.shape[0]+1-N_prediction), dtype = np.double)

    x[:, 0] = get_odometry_complete()
    x[:, 0] = [1,1,1,0,0,0,1,0,0,0,0,0,0]
    
    
    #TAREA DESEADA
    value = 15
    xd = lambda t: 4 * np.sin(value*0.04*t) + 3
    yd = lambda t: 4 * np.sin(value*0.08*t)
    zd = lambda t: 2 * np.sin(value*0.08*t) + 6
    xdp = lambda t: 4 * value * 0.04 * np.cos(value*0.04*t)
    ydp = lambda t: 4 * value * 0.08 * np.cos(value*0.08*t)
    zdp = lambda t: 2 * value * 0.08 * np.cos(value*0.08*t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)
    hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

    #quaternion = euler_to_quaternion(0, 0, psid) 
    quatd= np.zeros((4, t.shape[0]), dtype = np.double)


    # Calcular los cuaterniones utilizando la función euler_to_quaternion para cada psid
    for i in range(t.shape[0]):
        quaternion = euler_to_quaternion(0, 0, psid[i])  # Calcula el cuaternión para el ángulo de cabeceo en el instante i
        quatd[:, i] = quaternion  # Almacena el cuaternión en la columna i de 'quatd'


    # Reference Signal of the system
    xref = np.zeros((17, t.shape[0]), dtype = np.double)
    xref[0,:] = hxd         # px_d
    xref[1,:] = hyd         # py_d
    xref[2,:] = hzd         # pz_d 
    xref[3,:] = 0           # vx_d
    xref[4,:] = 0           # vy_d
    xref[5,:] = 0         # vz_d 
    xref[6,:] = quatd[0, :]         # qw_d
    xref[7,:] = quatd[1, :]         # qx_d
    xref[8,:] = quatd[2, :]        # qy_d
    xref[9,:] = quatd[3, :]         # qz_d
    xref[10,:] = 0         # wx_d
    xref[11,:] = 0         # wy_d
    xref[12,:] = 0         # wz_d
    

    # Initial Control values
    u_control = np.zeros((4, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 5
    phi_max = 0.5
    theta_max = 0.5
    psi_p_ref_max = 2
    

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_p_ref_min = -psi_p_ref_max

    # Create Optimal problem
    model, f, f_x, f_g = f_system_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((nx, N_prediction+1))
    simU = np.ndarray((nu, N_prediction))

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    # Simulation System

    print("Ready!!!")
       
    ############# EVAL RESTRICCIONES
    # Supongamos que ocp ya está creado y que ocp.model.con_h_expr contiene tu restricción
    constraint_expr = ocp.model.con_h_expr

    # Crear la función de CasADi para evaluar la restricción
    constraint_func = Function('constraint_func', [ocp.model.x, ocp.model.u], [constraint_expr])

    

    for k in range(0, t.shape[0]-N_prediction):
        tic = time.time()

        # Evaluar la restricción con valores actuales de x y u
        x_val = x[:, k]  # Valores actuales de las variables de estado
        u_val = u_control[:, k]  # Valores actuales de las variables de control

        constraint_value = constraint_func(x_val, u_val)
        

        # Almacenar el valor en el vector
        CBF_value[:, k] = (constraint_value[1])
        
        #print(CBF_value[:, k])

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

               # SET REFERENCES
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p", yref)

        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p", yref_N)

        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction')
        publish_matrix(xref[0:3, 0:500:5], '/task_desired')

        #time.sleep(1)

        #print(simX[:,10])

        u_control[:, k] = simU[:,0]

        # Get Computational Time
        status = acados_ocp_solver.solve()

        toc_solver = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")

        #u_control[:, k] = [9.81, 0.0000   ,0.0000      , 0.00000]
                
        # Send Control values
        send_velocity_control(u_control[:, k])

        #Imprime El vector de estados
        


        #print_state_vector(x[:, k])


        
        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_complete()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            
            send_full_state_to_sim(x[:, k+1])
        else:
            print("Opción no válida")

          
 
        time_module.sleep(1/frec)
        toc = time.time() - tic 
        print(1/toc)
    
    
    send_velocity_control([0, 0, 0, 0])

    print("Generating figures...")
    fig1 = plot_pose(x, xref, t)
    print("Saving figure 1_pose.png...")
    fig1.savefig("1_pose.png")
    print("✓ Saved 1_pose.png")
    
    
    fig3 = plot_vel_lineal(x[3:6,:], t)
    print("Saving figure 3_vel_lineal.png...")
    fig3.savefig("3_vel_lineal.png")
    print("✓ Saved 3_vel_lineal.png")
    
    fig4 = plot_vel_angular(x[10:13,:], t)
    print("Saving figure 4_vel_angular.png...")
    fig4.savefig("4_vel_angular.png")
    print("✓ Saved 4_vel_angular.png")

    fig5 = plot_CBF(CBF_value, t)
    print("Saving figure 5_CBF.png...")
    fig5.savefig("5_CBF.png")
    print("✓ Saved 5_CBF.png")


    fig2 = plot_control(u_control, t)
    print("Saving figure 2_control_actions.png...")
    fig2.savefig("2_control_actions.png")
    print("✓ Saved 2_control_actions.png")
    fig2.savefig("2_control_actions.png")
    

    

    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Local execution without ROS
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        send_velocity_control([0, 0, 0, 0])
    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
    else:
        print("Complete Execution")