import numpy as np
import matplotlib.pyplot as plt
import time


def gradient_method(f, gf, lsearch, x0, epsilon=1e-5):
    x = x0
    fs = []
    gs = []
    ts = []
    gval = gf(x)
    iter = 0
    start_Time = time.time()
    while np.linalg.norm(gval) >= epsilon:
        ts.append((time.time() - start_Time))
        t = lsearch(f, x, gf(x))
        x = x - t * gval
        # print('iter= {:2d} f(x)={:10.10f}'.format(ts[iter], f(x)))
        print(ts[iter])
        gval = gf(x)
        fs.append(f(x))
        gs.append(np.linalg.norm(gval))
        iter += 1

    return x, fs, gs, ts


def const_step(s):
    def lsearch(f, x, gf):
        return s

    return lsearch


def exact_quad(A):
    def lsearch(f, x, gf):
        np.linalg.cholesky(A)
        t = (np.linalg.norm(gf) ** 2) / (2 * np.linalg.norm(A @ gf) ** 2)
        return t

    return lsearch


def back(alpha, beta, s):
    def lsearch(f, xk, gk):
        t = s
        while f(xk) - f(xk - t * gk) < alpha * t * np.linalg.norm(gk) ** 2:
            t = t * beta
        return t

    return lsearch
def S_D(D):
        def S(X):
            sum = 0
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    sum += (np.linalg.norm(X[i]-X[j])**2 - D[i][j]**2 ) **2
            return 0.5 *sum
        return S

def G_S_D(D):
    def S_G(X):
        vec_lst = []
        for i in range(X.shape[0]):
            sum_x_i = 0
            for j in range(X.shape[0]):
                sum_x_i = (X[i]-X[j])*(np.linalg.norm(X[i]-X[j])**2 - D[i][j]**2)
            vec_lst.append(4*sum_x_i)
        return np.array(vec_lst)
    return S_G

def main():
    A = np.array(
        [[100, 2, 3, 4, 5], [6, 100, 8, 9, 10], [11, 12, 100, 14, 15], [16, 17, 18, 100, 20], [21, 22, 23, 24, 100]])
    max_eign = np.amax(np.linalg.eigvals(A.T @ A))
    s = 1 / (2 * max_eign)
    f = lambda x: x.T @ A.T @ A @ x
    gf = lambda x: 2 * A.T @ A @ x
    x0 = np.ones(5)
    eps = 10 ** -5
    # x_const, f_s_const, g_s_const, t_s_const = gradient_method(f, gf, const_step(s), x0, eps)
    # x_const_exact, f_s_exact, g_s_exact, t_s_exact = gradient_method(f, gf, exact_quad(A), x0, eps)
    # x_const_back, f_s_back, g_s_back, t_s_back = gradient_method(f, gf, back(0.5, 0.5, 1), x0, eps)
    # ######graph - fvalue ############
    # p1 = plt.loglog(np.arange(1, len(f_s_const) + 1), f_s_const, label="const")
    # plt.xlim(left=1)
    # plt.title('Gradient descend - f value')
    # p2 = plt.loglog(np.arange(1, len(f_s_exact) + 1), f_s_exact, label="exact")
    # plt.xlim(left=1)
    # p3 = plt.loglog(np.arange(1, len(f_s_back) + 1), f_s_back, label="back")
    # plt.xlim(left=1)
    # plt.legend((p1[0], p2[0], p3[0]), ('const', 'exact quad', 'back'))
    # plt.show()
    # ######graph - gnorm ############
    # p1 = plt.loglog(np.arange(1, len(g_s_const) + 1), g_s_const, label="const")
    # plt.xlim(left=1)
    # plt.title('Gradient descend - g norm')
    # p2 = plt.loglog(np.arange(1, len(g_s_exact) + 1), g_s_exact, label="exact")
    # plt.xlim(left=1)
    # p3 = plt.loglog(np.arange(1, len(g_s_back) + 1), g_s_back, label="back")
    # plt.xlim(left=1)
    # plt.legend((p1[0], p2[0], p3[0]), ('const', 'exact quad', 'back'))
    # plt.show()
    # ######graph - gnorm -TIME ############
    # p1 = plt.semilogy(t_s_const, g_s_const, label="const")
    # plt.title('Gradient descend - gf value per mili second')
    # p2 = plt.semilogy(t_s_exact, g_s_exact, label="exact")
    # p3 = plt.semilogy(t_s_back, g_s_back, label="const")
    # plt.legend((p1[0], p2[0], p3[0]), ('const', 'exact quad', 'back'))
    # plt.show()
    ##############Q2#####################
    B = np.array(
        [[1, 0, 0, -1, -1, -1, -1, -1, 0], [0, 1, -1, -1, -1, -1, -1, -1, -1], [1, 1, -1, -1, -1, -1, -1, -1, -1],
         [0, 0, -1, 1, -1, -1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, 1, 0], [-1, -1, 1, 1, 1, 0, 1, 0, 1],
         [-1, -1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, 1, 1, 0, 1, 1, 0, 0]])
    x_const, f_s_const, g_s_const, t_s_const = gradient_method(f, gf, const_step(s), x0, eps)
    D = np.zeros((B.shape[0], B.shape[0]))
    for i in range(B.shape[0]):
        for j in range(B.shape[0]):
            D[i, j] = np.linalg.norm(B[:j] - B[:i])

        if __name__ == "__main__":
            main()
    S(x)= lambda x: x.T @ A.T @ A @ x