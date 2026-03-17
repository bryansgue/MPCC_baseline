import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

# Definir la curva en 3D y su derivada
def r(t):
    """ Devuelve el punto en la curva para el parámetro t. """
    return np.array([np.sin(t), np.cos(t), t])

def r_prime(t):
    """ Devuelve la derivada de la curva en el parámetro t. """
    return np.array([np.cos(t), -np.sin(t), 1])

def integrand(t):
    """ Devuelve la norma de la derivada de la curva en el parámetro t. """
    return np.linalg.norm(r_prime(t))

# Calcular la longitud de arco acumulada desde t0 hasta tk
def arc_length(tk, t0=0):
    """ Calcula la longitud de arco desde t0 hasta tk. """
    length, _ = quad(integrand, t0, tk)
    return length

# Encontrar t correspondiente a una longitud de arco dada
def find_t_for_length(theta, t0=0, t_max=10):
    """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
    func = lambda t: arc_length(t, t0) - theta
    return bisect(func, t0, t_max)

# Convertir una longitud de arco a punto en la curva
def length_to_point(theta, t0=0, t_max=10):
    """ Convierte una longitud de arco theta a un punto en la curva. """
    tk = find_t_for_length(theta, t0, t_max)
    return r(tk)

# Calcular el error de posición
def position_error(tk):
    """ Calcula el error de posición entre el punto en t_k y la posición deseada. """
    # Calcular la longitud de arco en el instante t_k
    theta_k = arc_length(tk)
    
    # Obtener la posición actual en t_k
    p_k = r(tk)
    
    # Obtener la posición deseada en la longitud de arco theta_k
    p_d_theta_k = length_to_point(theta_k)
    
    # Calcular el error
    error_p = p_k - p_d_theta_k
    return error_p

# Parámetro de tiempo k
t_k = 3 # Tiempo actual
error = position_error(t_k)
print("Error de posición en t =", t_k, "es", error)
