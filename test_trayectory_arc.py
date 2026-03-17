import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

# Definir el valor global
value = 21

def trayectoria(t):
    """ Crea y retorna las funciones para la trayectoria y sus derivadas. """
    def xd(t):
        return 4 * np.sin(value * 0.04 * t) + 3

    def yd(t):
        return 4 * np.sin(value * 0.08 * t)

    def zd(t):
        return 2 * np.sin(value * 0.08 * t) + 6

    def xd_p(t):
        return 4 * value * 0.04 * np.cos(value * 0.04 * t)

    def yd_p(t):
        return 4 * value * 0.08 * np.cos(value * 0.08 * t)

    def zd_p(t):
        return 2 * value * 0.08 * np.cos(value * 0.08 * t)

    return xd, yd, zd, xd_p, yd_p, zd_p

def r(t, xd, yd, zd):
    """ Devuelve el punto en la trayectoria para el parámetro t usando las funciones de trayectoria. """
    return np.array([xd(t), yd(t), zd(t)])

def r_prime(t, xd_p, yd_p, zd_p):
    """ Devuelve la derivada de la trayectoria en el parámetro t usando las derivadas de las funciones de trayectoria. """
    return np.array([xd_p(t), yd_p(t), zd_p(t)])

def integrand(t, xd_p, yd_p, zd_p):
    """ Devuelve la norma de la derivada de la trayectoria en el parámetro t. """
    return np.linalg.norm(r_prime(t, xd_p, yd_p, zd_p))

def arc_length(tk, t0=0, xd_p=None, yd_p=None, zd_p=None):
    """ Calcula la longitud de arco desde t0 hasta tk usando las derivadas de la trayectoria. """
    length, _ = quad(integrand, t0, tk, args=(xd_p, yd_p, zd_p))
    return length

def find_t_for_length(theta, t0=0, t_max=None, xd_p=None, yd_p=None, zd_p=None):
    """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
    func = lambda t: arc_length(t, t0, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p) - theta

    # Verificar los valores iniciales
    f_t0 = func(t0)
    f_tmax = func(t_max)

    if f_t0 * f_tmax > 0:
        print(f"Advertencia: func(t0) y func(t_max) tienen el mismo signo. Intentando ajustar t_max.")
        # Ajustar dinámicamente t_max
        t_max_adjusted = t_max
        step = (t_max - t0) / 10
        while f_t0 * f_tmax > 0 and t_max_adjusted > t0:
            t_max_adjusted -= step
            f_tmax = func(t_max_adjusted)
            if f_t0 * f_tmax <= 0:
                t_max = t_max_adjusted
                break

        if f_t0 * f_tmax > 0:
            raise ValueError("No se pudo encontrar un intervalo adecuado para la bisección.")

    # Intentar la bisección con el nuevo intervalo
    try:
        tk = bisect(func, t0, t_max)
        return tk
    except ValueError as e:
        print(f"Error en bisect: {e}")
        raise ValueError("No se pudo encontrar un valor de t que corresponda a la longitud de arco theta.")


def length_to_point(theta, t0=0, t_max=None, xd=None, yd=None, zd=None, xd_p=None, yd_p=None, zd_p=None):
    """ Convierte una longitud de arco theta a un punto en la trayectoria. """
    tk = find_t_for_length(theta, t0, t_max, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)
    return r(tk, xd, yd, zd)

def calculate_positions_in_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):
    """ Calcula los puntos en la trayectoria y la longitud de arco para cada instante en t_range. """
    positions = []
    arc_lengths = []
    
    # Generar los valores de longitud de arco y las posiciones correspondientes
    for tk in t_range:
        theta = arc_length(tk, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)
        arc_lengths.append(theta)
        point = length_to_point(theta, t_max=t_max, xd=xd, yd=yd, zd=zd, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)
        positions.append(point)

    # Crear la función que retorna la posición dado un valor de longitud de arco
    def position_by_arc_length(s):
        return length_to_point(s, t_max=t_max, xd=xd, yd=yd, zd=zd, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)

    return np.array(arc_lengths), np.array(positions).T, position_by_arc_length

def calculate_orthogonal_error(error_total, tangent):

    if np.linalg.norm(tangent) == 0:
        return error_total  # No hay tangente válida, devolver el error total
    # Matriz de proyección ortogonal
    I = np.eye(3)  # Matriz identidad en 3D
    P_ec = I - np.outer(tangent, tangent)
    # Aplicar la matriz de proyección para obtener el error ortogonal
    e_c = P_ec @ error_total
    return e_c

def main():
    # Definir el rango de instantes
    #t = np.arange(0, 10, 0.1)  # Ajusta el rango y el paso según sea necesario

    t_final = 10
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = 30
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Obtener las funciones de trayectoria y sus derivadas
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)

    # Calcular la trayectoria deseada en el rango de t
    ref = np.array([r(ti, xd, yd, zd) for ti in t]).T

    # Inicializar xref
    xref = np.zeros((3, t.shape[0]), dtype=np.double)
    xref[0, :] = ref[0, :]  # px_d
    xref[1, :] = ref[1, :]  # py_d
    xref[2, :] = ref[2, :]  # pz_d 

    # Calcular posiciones parametrizadas en longitud de arco
  #positions = calculate_positions_in_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t, t_max=t_final)
    arc_lengths, pos_ref, position_by_arc_length = calculate_positions_in_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t, t_max=t_final)

    

    print("HERE")
    # Inicializar el array para almacenar v_theta
    v_theta = np.zeros(len(t))

    for k in range(len(t)):
        # Calcular el error de posición para cada instante k
        # Calcular el cambio en longitud de arco usando positions

        s_k = arc_lengths[k]

        delta_theta_k = np.linalg.norm(pos_ref[:, k] - pos_ref[:, k-1])
        
        # Calcular v_theta
        v_theta[k] = delta_theta_k / t_s

        # Calcular el error de posición para cada instante k
        error = xref[:, k] - pos_ref[:, k]

        # Vector tangente (dirección)
        tangent = (pos_ref[:, k] - pos_ref[:, k-1]) / delta_theta_k

        

        # Proyección del error en la dirección del tangente
        error_arrastre = np.dot(tangent, error) * tangent

        # Cálculo del error ortogonal (error de contorno)
        error_contorno = calculate_orthogonal_error(error, tangent)


        # Imprimir el error de contorno para cada instante
        #print(f"t = {t[k]:.1f}, error de contorno: {error_contorno }")
        #print(f"t = {t[k]:.1f}, error_arrastre: {error_arrastre }")

        
        print(error)
      
        

if __name__ == "__main__":
    main()
