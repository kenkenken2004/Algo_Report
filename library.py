import numpy as np
from typing import Callable
from scipy.ndimage import convolve


def laplacian(matrix: np.ndarray, dx):
    # Define the kernel for 4th order central difference in each direction
    kernel = [
        [0, 0, -1, 0, 0],
        [0, 0, 16, 0, 0],
        [-1, 16, -30 - 30, 16, -1],
        [0, 0, 16, 0, 0],
        [0, 0, -1, 0, 0]
    ]
    # Convolve the kernel with the function 'f' in both x and y directions
    l = convolve(matrix, kernel, mode='reflect')
    l = l / (12 * dx ** 2)
    return l


def runge_kutta_4th_order(matrix: np.ndarray, dt: float, dx: float,
                          func: Callable[[[float], np.ndarray, np.ndarray], np.ndarray]):
    L1 = laplacian(matrix[:, :, 0], dx)
    L2 = laplacian(matrix[:, :, 1], dx)
    L3 = laplacian(matrix[:, :, 2], dx)

    def step(v1: float, v2: float, v3: float, l1: float, l2: float, l3: float):
        v = np.array([v1, v2, v3])
        _l = np.array([l1, l2, l3])
        k1: np.ndarray = func(dt, v, _l)
        k2: np.ndarray = func(dt, v + dt / 2 * k1, _l)
        k3: np.ndarray = func(dt, v + dt / 2 * k2, _l)
        k4: np.ndarray = func(dt, v + dt * k3, _l)
        slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        nextValue: float = v + dt * 2 * slope
        return nextValue

    result: np.ndarray = np.array(np.frompyfunc(step, 6, 1)(matrix[:, :, 0], matrix[:, :, 1], matrix[:, :, 2], L1,
                                                            L2, L3))
    print()
    return result
