import math
import random
import numpy as np


eps = 1e-5
n = 200
m = 100  # number of monte carlo samples for mcem
phi_true = [0.8, -0.2]


def random_init():
    return -1 + 2 * random.random()


def get_mean(idx, x):
    if idx == 1:
        return -1 + x
    elif idx == 2:
        return -2 + 0.5 * math.exp(0.5 + x)
    elif idx == 3:
        return -1 + math.sin(2 * x)
    elif idx == 4:
        return -1 + 0.4 * x * x * x
    else:
        raise Exception('In "get_mean" function, wrong idx is given: ' + str(idx))


def get_error(idx, x=None):
    if idx == 1:
        return np.random.normal(0, math.sqrt(0.9))
    elif idx == 2:
        return np.random.normal(0, math.sqrt(0.49 * (1 + x * x)))
    elif idx == 3:
        return np.random.lognormal(-0.245, 0.7)
    else:
        raise Exception('In "get_error" function, wrong idx is given: ' + str(idx))


def get_pi(phi, v):
    try:
        return 1 / (1 + math.exp(-phi[0] - phi[1] * v))
    except OverflowError:
        print('overflow: ' + str(-phi[0] - phi[1] * v))
        return 1e-10


def get_delta(y):
    return np.random.binomial(1, get_pi(phi_true, y))


def generate_data(error_model, mean_structure):
    X, Y, D = [], [], []
    for _ in range(n):
        x = np.random.normal(0, math.sqrt(0.5))
        y = get_mean(mean_structure, x) + get_error(error_model, x)
        d = get_delta(y)
        X.append(x)
        Y.append(y)
        D.append(d)
    return X, Y, D


def sample_for_mcem(x, theta):
    beta0, beta1, sigma_sq = theta
    while True:
        y = np.random.normal(beta0 + beta1 * x, math.sqrt(sigma_sq))
        if get_delta(y) == 0:
            return y


