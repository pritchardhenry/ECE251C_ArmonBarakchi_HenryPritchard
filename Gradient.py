import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def p_norm(x, p):
    x = np.array(x, dtype=float)
    return np.sum(np.abs(x)**p)**(1/p)

def deriv_phi(x, beta, M):
    return (-1)*sigmoid(beta*(x-M))

def deriv_g(x, p, pixel_k = (0,0), ):


def calcGradient ( h, pixel_k = (0,0), )