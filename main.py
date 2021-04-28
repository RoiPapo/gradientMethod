import numpy as np


def gradient_method(f, gf, lsearch, x0, epsilon=1e-5):
    x = x0
    fs = []
    gs = []
    gval = gf(x)
    iter = 0
    while np.linalg.norm(gval) >= epsilon:
        iter += 1
        t = lsearch(f, x, gf(x))
        x = x - t * gval
        print('iter= {:2d} f(x)={:10.10f}'.format(iter, f(x)))
        gval = gf(x)
        fs.append(f(x))
        gs.append(np.linalg.norm(gval))

    return x


def const_step(s):
    def lsearch(f, x, gf):
        return s

    return lsearch


def main():
    f = lambda x: x ** 2
    gf = lambda x: 2 * x
    x0 = 50
    gradient_method(f, gf, const_step(0.1), x0)


if __name__ == "__main__":
    main()
