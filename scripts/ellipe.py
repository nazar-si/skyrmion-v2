import numpy as np
import numba as nb # type: ignore
import scipy.special as sp

@nb.jit
def Q(x):
    return 2.50000000e-01 + \
        x * 9.37499997e-02 + \
        x ** 2 * 5.85936634e-02 + \
        x ** 3 * 4.27180927e-02 + \
        x ** 4 * 3.34833905e-02 + \
        x ** 5 * 2.61769742e-02 + \
        x ** 6 * 1.68862164e-02 + \
        x ** 7 * 6.50609490e-03 + \
        x ** 8 * 1.00962793e-03 + \
        x ** 9 * 3.27954899e-05
@nb.jit
def P(x):
    return 1.00000000e+00 + \
        x * 4.43147181e-01 + \
        x ** 2 * 5.68051946e-02 + \
        x ** 3 * 2.18317996e-02 + \
        x ** 4 * 1.15688437e-02 + \
        x ** 5 * 7.58395289e-03 + \
        x ** 6 * 7.77395493e-03 + \
        x ** 7 * 1.07350949e-02 + \
        x ** 8 * 8.68786817e-03 + \
        x ** 9 * 2.50888492e-03 + \
        x ** 10 * 1.53552577e-04

@nb.jit
def Q_x(x):
    return 9.37499997e-02 + \
        2 * x * 5.85936634e-02 + \
        3 * x ** 2 * 4.27180927e-02 + \
        4 * x ** 3 * 3.34833905e-02 + \
        5 * x ** 4 * 2.61769742e-02 + \
        6 * x ** 5 * 1.68862164e-02 + \
        7 * x ** 6 * 6.50609490e-03 + \
        8 * x ** 7 * 1.00962793e-03 + \
        9 * x ** 8 * 3.27954899e-05
@nb.jit
def P_x(x):
    return 4.43147181e-01 + \
        2 * x * 5.68051946e-02 + \
        3 * x ** 2 * 2.18317996e-02 + \
        4 * x ** 3 * 1.15688437e-02 + \
        5 * x ** 4 * 7.58395289e-03 + \
        6 * x ** 5 * 7.77395493e-03 + \
        7 * x ** 6 * 1.07350949e-02 + \
        8 * x ** 7 * 8.68786817e-03 + \
        9 * x ** 8 * 2.50888492e-03 + \
        10 * x ** 9 * 1.53552577e-04

@nb.jit
def Pk(x):
    return x ** 10 * 1.37982864606273237150E-4 + \
            x ** 9 * 2.28025724005875567385E-3 + \
            x ** 8 * 7.97404013220415179367E-3 + \
            x ** 7 * 9.85821379021226008714E-3 + \
            x ** 6 * 6.87489687449949877925E-3 + \
            x ** 5 * 6.18901033637687613229E-3 + \
            x ** 4 * 8.79078273952743772254E-3 + \
            x ** 3 * 1.49380448916805252718E-2 + \
            x ** 2 * 3.08851465246711995998E-2 + \
            x * 9.65735902811690126535E-2 + \
            1.38629436111989062502E0

@nb.jit
def Qk(x):
    return x ** 10 * 2.94078955048598507511E-5 + \
            x ** 9 * 9.14184723865917226571E-4 + \
            x ** 8 * 5.94058303753167793257E-3 + \
            x ** 7 * 1.54850516649762399335E-2 + \
            x ** 6 * 2.39089602715924892727E-2 + \
            x ** 5 * 3.01204715227604046988E-2 + \
            x ** 4 * 3.73774314173823228969E-2 + \
            x ** 3 * 4.88280347570998239232E-2 + \
            x ** 2 * 7.03124996963957469739E-2 + \
            x * 1.24999999999870820058E-1 + \
            4.99999999999999999821E-1

Ck = 1.3862943611198906188E0

@nb.jit
def ellipe(x: float):
    if (x == 0):
        return np.pi/2
    if (x == 1):
        return 1.0
    if (x > 0):
        z = 1 - x
        return P(z) - z * np.log(z) * Q(z)
    z = - 1 / (x - 1)
    return (P(z) - z * np.log(z) * Q(z)) * np.sqrt(1 - x)

@nb.jit
def ellipk(x: float):
    if (x == 0):
        return 0
    if (x == 1):
        return np.pi/2
    if x > 1:
        return (Pk(1/x) - np.log(1/x) * Qk(1/x)) / np.sqrt(x)
    if (x < 1e-12):
        return Ck - 0.5 * np.log(x)
    if (x > 0):
        return Pk(x) - np.log(x) * Qk(x)

@nb.jit 
def ellipe_x(x: float):
    if (x >= 0):
        z = 1 - x
        return - P_x(z) + np.log(z) * Q(z) + Q(z) + z * np.log(z) * Q_x(z)
    z = - 1 / (x - 1)
    return (P_x(z) - np.log(z) * Q(z) - Q(z) - z * np.log(z) * Q_x(z)) * z ** 2 * np.sqrt(1 - x) - (P(z) - z * np.log(z) * Q(z)) / np.sqrt(1 - x) / 2


if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    x = np.linspace(0, 2, 1000)

    ellipk = np.vectorize(ellipk)
    plt.plot(x, ellipk(x))
    plt.plot(x, sp.ellipk(x))
    plt.show()
    # ellipe_x = np.vectorize(ellipe_x)
    # plt.plot(x, (ellipe(x + 0.001) - ellipe(x)) / 0.001)
    # plt.plot(x, ellipe_x(x))
    # plt.show()