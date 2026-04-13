import numpy as np 
import matplotlib.pyplot as plt 

from scripts.read_field import chirality, draw_field


def generate(eta):
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))

    r, w = 1, 0.75
    R = np.sqrt(X ** 2 + Y ** 2)
    THETA = 2 * np.arctan(np.sinh(r/w) / np.sinh(R/w))
    PHI = np.atan2(Y, X) + eta

    MX = np.sin(THETA) * np.cos(PHI)
    MY = np.sin(THETA) * np.sin(PHI)
    MZ = np.cos(THETA)

    M = np.stack([MX, MY, MZ]).transpose((1,2,0))
    return M

fig, axs = plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        M = generate((i + j * 3) / 8 * np.pi)
        ax = axs[j][i]
        ax.set_title(f"$\\eta = {i + j *3 } \\cdot \\pi / 8 $")
        draw_field(ax, M, mode='spirality')
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()