import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt


def calculate_acceleration(road):
    # Parameters
    reference_distance = 7
    T = 0.05  # Sampling time
    Ts = 0.1  # Inertia time constant

    # System matrices
    A = np.array([[1, T, 0],
                  [0, 1, T],
                  [0, 0, 1 - T / Ts]])

    B = np.array([[0],
                  [0],
                  [T / Ts]])

    # LQR cost matrices
    Q = np.diag([1, 1, 1])  # State cost matrix
    # 另一个对称正定的矩阵
    R = np.array([1])  # 一个对角元素为正的对称矩阵
    # Control cost matrix
    P = solve_discrete_are(A, B, Q, R)
    # authority = np.array([[0.1, 0.1],
    #                       [1.5, 1.5]])
    authority = np.array([[2, 2],
                          [2, 2]])

    # Compute the LQR gain
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    # Calculate state variables
    v0 = road.vehicles[0].speed
    v1 = road.vehicles[1].speed
    v2 = road.vehicles[2].speed
    x0 = road.vehicles[0].position[0]
    x1 = road.vehicles[1].position[0]
    x2 = road.vehicles[2].position[0]
    a0 = road.vehicles[0].action["acceleration"]
    a1 = road.vehicles[1].action["acceleration"]
    a2 = road.vehicles[2].action["acceleration"]

    delta_d1 = reference_distance - (x2 - x1)
    delta_v1 = v1 - v2
    delta_a1 = a1 - a2

    delta_d2 = reference_distance - (x1 - x0)
    delta_v2 = v0 - v1
    delta_a2 = a0 - a1

    big_delta_d1 = reference_distance*1 - (x2 - x1)
    big_delta_v1 = v1 - v2
    big_delta_a1 = a1 - a2

    big_delta_d2 = reference_distance*2 - (x2 - x0)
    big_delta_v2 = v0 - v2
    big_delta_a2 = a0 - a2

    # State vector
    X1 = np.array([delta_d1, delta_v1, delta_a1])
    Y1 = np.array([big_delta_d1, big_delta_v1, big_delta_a1])
    X2 = np.array([delta_d2, delta_v2, delta_a2])
    Y2 = np.array([big_delta_d2, big_delta_v2, big_delta_a2])

    # Control input (acceleration)
    u1 = -K @ X1
    w1 = -K @ Y1
    b1 = [u1[0], w1[0]]
    a1 = authority[0][0]*b1[0]+authority[0][1]*b1[1]

    u2 = -K @ X2
    w2 = -K @ Y2
    b2 = [u2[0], w2[0]]
    a0 = authority[0][0]*b2[0]+authority[0][1]*b2[1]

    return [a0, a1]