from copy import deepcopy
from data import *
from scipy.optimize import fsolve as solve

# from scipy.optimize import newton_krylov as solve
# from scipy.optimize import broyden2 as solve

solver_cnt = 5


def get_mle_of_theta(phi_mle, Y, D):
    numer, denom = 0, 0
    for i in range(n):
        if D[i] == 1:
            t = 1 / get_pi(phi_mle, Y[i])
            numer += t * Y[i]
            denom += t
    return numer / denom


def full_sample(Y):
    # print('full_sample')
    return np.average(Y)


def missing_at_random(X, Y, D):
    # print('missing_at_random')
    def score_of_phi(phi):
        res = [0, 0]
        for i in range(n):
            t = D[i] - get_pi(phi, X[i])
            res[0] += t
            res[1] += t * X[i]
        return res

    phi_init = np.array([random_init(), random_init()])
    phi_mle = solve(score_of_phi, phi_init)
    return get_mle_of_theta(phi_mle, Y, D)


def fully_parametric(X, Y, D):
    # print('fully_parametric')
    # Here we use MCEM.
    def score_of_beta_for_mcem(beta):
        res = [0, 0]
        for j in range(m):
            for i in range(n):
                t = Y_new[j][i] - beta[0] - beta[1] * X[i]
                res[0] += t
                res[1] += t * X[i]
        return res

    def get_sigma_sq(beta):
        res = 0
        for j in range(m):
            for i in range(n):
                t = Y_new[j][i] - beta[0] - beta[1] * X[i]
                res += t * t
        return res / (n * m)

    def is_end(cur, prev):
        diff = 0
        for i in range(3):
            t = cur[i] - prev[i]
            diff += t * t
        return diff < eps * 100

    eta = [random_init(), random_init(), abs(random_init())]
    eta_prev = [eta[0] - 1, eta[1] - 1, eta[2] + 1]

    while not is_end(eta, eta_prev):
        eta_prev = deepcopy(eta)
        Y_new = [[Y[i] if D[i] == 1
                  else sample_for_mcem(X[i], eta_prev)
                  for i in range(n)] for _ in range(m)]
        beta_prev = np.array([eta_prev[0], eta_prev[1]])
        beta = solve(score_of_beta_for_mcem, beta_prev)
        sigma_sq = get_sigma_sq(beta)
        eta = [beta[0], beta[1], sigma_sq]

    beta = eta[:2]
    return np.average([Y[i] if D[i] == 1
                       else (beta[0] + beta[1] * X[i])
                       for i in range(n)])


def gaussian_mixture(X, Y, D):
    # print('gaussian_mixture')

    def score_of_phi_ck(phi):
        res = [0, 0]
        for i in range(n):
            t = D[i] / get_pi(phi, Y[i]) - 1
            res[0] += t
            res[1] += t * X[i]
        return res

    phi_init = np.array([random_init(), random_init()])
    phi_mle = solve(score_of_phi_ck, phi_init)
    return get_mle_of_theta(phi_mle, Y, D)


def new_method(X, Y, D):
    # print('new_method')
    beta = [0, 0]

    def score(d, y, phi):
        t = d - get_pi(phi, y)
        return t, t * y

    def odds(y, phi):
        pi = get_pi(phi, y)
        return (1 - pi) / pi

    def score_of_phi_for_em(phi):
        res = [0, 0]
        for i in range(n):
            if D[i] == 1:
                s0, s1 = score(D[i], Y[i], phi)
                res[0] += s0
                res[1] += s1
            else:
                w_sum = 0
                ans = [0, 0]
                for j in range(n):
                    if D[j] == 1:
                        w = W[i][j]
                        w_sum += w
                        s0, s1 = score(D[i], Y[j], phi)
                        ans[0] += w * s0
                        ans[1] += w * s1
                res[0] += ans[0] / w_sum
                res[1] += ans[1] / w_sum
        return res

    def score_of_beta(beta):
        res = [0, 0]
        for i in range(n):
            if D[i] == 1:
                t = Y[i] - beta[0] - beta[1] * X[i]
                res[0] += t
                res[1] += t * X[i]
        return res

    def get_sigma_sq(beta):
        r = 0
        res = 0
        for i in range(n):
            if D[i] == 1:
                t = Y[i] - beta[0] - beta[1] * X[i]
                res += t * t
                r += 1
        return res / r

    def get_diff(cur, prev):
        diff = 0
        for i in range(2):
            t = cur[i] - prev[i]
            diff += t * t
        # print(diff)
        return diff
        # return diff < 2e-2

    my_iter = 0
    while True:
        beta_init = np.array([random_init(), random_init()])
        if my_iter < 10:
            try:
                beta = solve(score_of_beta, beta_init)
                break
            except RuntimeWarning:
                print('No Convergence Error! in new method')
            my_iter += 1
        else:
            beta = solve(score_of_beta, beta_init)
            break

    # print('beta: ' + str(beta))
    sigma_sq = get_sigma_sq(beta)
    # print('sigma_sq:' + str(sigma_sq))

    def f1(i, j):
        t = Y[j] - beta[0] - beta[1] * X[i]
        return math.exp(-t * t / (2 * sigma_sq)) / math.sqrt(2 * math.pi * sigma_sq)

    def coeff(j):
        ans = 0
        for l in range(n):
            if D[l] == 1:
                ans += f1(l, j)
        return ans

    C = [coeff(i) if D[i] == 1 else 1 for i in range(n)]
    W_base = [[0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            if D[j] == 1:
                W_base[i][j] = f1(i, j) / C[j]

    # phi = np.array([random_init(), random_init()])
    phi = np.array(deepcopy(phi_true))

    min_diff = 100
    phi_best = deepcopy(phi)
    my_iter = 0
    while True:
        phi_prev = deepcopy(phi)
        W = [[odds(Y[j], phi_prev) * W_base[i][j] for j in range(n)] for i in range(n)]
        phi = solve(score_of_phi_for_em, np.array(phi_prev))

        diff = get_diff(phi, phi_prev)
        if min_diff > diff:
            phi_best = deepcopy(phi)
            min_diff = diff
            my_iter = 0
        else:
            my_iter += 1
        if diff < eps or my_iter > 10:
            break

    # print('phi_best: ' + str(phi_best))
    return get_mle_of_theta(phi_best, Y, D)








