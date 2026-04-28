import discretisedfield as df
import os 
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Literal
from cmap import Colormap, Color
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb


def file_to_array(filename: str, scaling=1e5):
    read_field = df.Field.from_file(filename)
    field = read_field.sel("z").array.transpose((1,0,2)) /scaling
    field = field[::-1,:,:]
    field[:,:,1] = -field[:,:,1]
    return field

def _resolve_folder(dir_name: str) -> str:
    """
    Accept either:
    - a direct path to a drive folder (recommended), e.g. 'test/skyrmion_system/drive-0-...'
    - a legacy name that exists under './skyrmion_system/<name>'
    """
    if os.path.isdir(dir_name):
        return dir_name

    legacy = os.path.join("./skyrmion_system", dir_name)
    if os.path.isdir(legacy):
        return legacy

    raise FileNotFoundError(f"Folder not found: {dir_name}")

def read_dir(dir_name: str, delete_files: bool = False, new_name = None, scaling=1e5):
    folder = _resolve_folder(dir_name)
    files = [file for file in os.listdir(folder) if file.endswith(".omf") and file.startswith("skyrmion_system-Oxs")]
    files.sort()
    array = []
    for file in files:
        try:
            data = file_to_array(os.path.join(folder, file), scaling)
            # np.save(f"./skyrmion_system/{dir_name}/magnetization.npy", data)
            array.append(data)
            if delete_files:
                os.remove(os.path.join(folder, file))
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    if (new_name is not None and new_name != dir_name):
        while os.path.exists(os.path.join(new_name)):
            new_name = f"{new_name}-2"
        os.rename(folder, os.path.join(new_name))
    return np.array(array)

def chirality(M):
    return np.gradient(M[:,:,2], axis=1) * M[:,:,0] + np.gradient(M[:,:,2], axis=0) * M[:,:,1]

def spirality(M):
    G = np.stack([np.gradient(M[:,:,2], axis=1), np.gradient(M[:,:,2], axis=0)])
    # normalized gradient in XY plane (d_x m_x, d_y m_y)
    G = G / np.clip(np.linalg.norm(G, axis=0, ord=2, keepdims=True), 1e-12, None)
    # $\chi$ is an angle of the magnetization in the plane perpendicular to it's gradient in XY plane  
    CHI = np.atan2(
        G[1] * M[:,:,0] - G[0] * M[:,:,1],
        M[:,:,2]
    )
    
    return CHI

# Функция дает распределению цвет в зависимости от знака киральности (знака антисимметричного обмена)
def chirality_signed_field(M, t = 0):
    return np.sign(chirality(M)) * (M[:, :, 2] + 1) / 2


_cmap = Colormap('colorcet:cet_cbtc2').to_mpl()
cmap = Colormap([_cmap(0), _cmap(0.25), _cmap(0.5), _cmap(0.75), _cmap(1)]).to_mpl()
cmap_chi = Colormap([_cmap(0), Color('orchid'), _cmap(1)]).to_mpl()
cmap2 = Colormap([cmap(0.5), Color('lightcoral'), cmap(0.5), Color('lightsteelblue'), cmap(0.5)]).to_mpl()
# cmap2 = Colormap('colorcet:cet_c3').to_mpl()

def draw_field(ax, M, mode: Literal['m_z', 'quiver', 'byside', 'chirality', 'normal', 'polar', 'spirality', 'rotation']='m_z', cutoff = 50, quiver_scale = 10, chirality_threshold = 0, rotate_to_average = False):
    length = M.shape[0]
    density = length * 0.025
    if (rotate_to_average):
        mean_y = np.mean((M[cutoff:-cutoff,cutoff:-cutoff,2] + 1) ** 2, axis = 1)
        mean_x = np.mean((M[cutoff:-cutoff,cutoff:-cutoff,2] + 1) ** 2, axis = 0)
        mx = -np.sum(np.linspace(-1, 1, mean_x.shape[0]) * mean_x) / np.sum(mean_x)
        my = np.sum(np.linspace(-1, 1, mean_y.shape[0]) * mean_y) / np.sum(mean_y)
        angle = np.atan2(my, mx) 
        IdX, IdY = np.meshgrid(np.arange(cutoff, length - cutoff) - length / 2, np.arange(cutoff, length - cutoff) - length / 2)
        IdX, IdY = np.cos(angle) * IdX + np.sin(angle) * IdY + length / 2, - np.sin(angle) * IdX + np.cos(angle) * IdY + length / 2
        IdX, IdY = np.minimum(np.maximum(IdX, 0), length - 1).astype(int), np.minimum(np.maximum(IdY, 0), length - 1 ).astype(int)
        C = gaussian_filter((np.tanh(chirality(M) * 100) + 1) / 2, sigma=10)[IdY, IdX]
        S = (np.sin(spirality(M)) ** 2)[IdY, IdX]
        M = M[IdY, IdX]
    else:
        M = M[cutoff:-cutoff, cutoff:-cutoff]
        C = gaussian_filter((np.atan(chirality(M) * 100) + 1) / 2, sigma=10)
        S = gaussian_filter(np.sin(spirality(M)) ** 2, sigma=20)

    size = (density - cutoff / density) / 2
    extent = [-size, size, -size, size]
    ax.set_xticks([-5,0,5])
    ax.set_yticks([-5,0,5])


    if mode == 'quiver':
        X, Y = np.meshgrid(np.arange(0, length - 2 * cutoff, quiver_scale), np.arange(0, length - 2 * cutoff, quiver_scale))
        U, V = M[Y,X, 0], M[Y,X,1]
        ax.quiver((X - length / 2 + cutoff) / (2 * density), (Y - length / 2 + cutoff) / (2 * density), U, V, width=0.005, scale=density * 2)

    if mode == 'normal':
        return ax.imshow(np.astype(127 * (M + 1), np.uint8), extent=extent)
    if mode == 'polar':
        m_z = M[:,:,2]
        m_z_x = np.gradient(m_z, axis=1)
        m_z_y = np.gradient(m_z, axis=0)
        m_x = M[:,:,0]
        m_y = M[:,:,1]

        mag = np.sqrt((m_x ** 2 + m_y ** 2) * (m_z_x ** 2 + m_z_y ** 2))
        sin = (m_x * m_z_y - m_y * m_z_x) / mag
        cos = (m_x * m_z_x + m_y * m_z_y) / mag

        S = (1 - M[:,:,2] ** 2) ** 10
        V = (1 + M[:,:,2]) / 2 * 0.3 + 0.7
        H = (np.arctan2(M[:,:,1],-M[:,:,0]) + np.pi) / (2 * np.pi)
        # H = (np.arctan2(sin, cos) + np.pi) / (2 * np.pi)
        return ax.imshow(np.astype(255 * hsv_to_rgb(np.stack([H,S,V]).transpose((1,2,0))), np.uint8), extent=extent)
    if mode == 'm_z':
        return ax.imshow(M[:,:,2], cmap=cmap2, vmin=-1, vmax=1, extent=extent)
    if mode == 'byside':
        ax.set_xticks([-size * 2 - 5, -size * 2, - size * 2 + 5, -5, 0, 5, size * 2 - 5, size * 2, size * 2 + 5])
        ax.set_xticklabels([-5, 0, 5, -5, 0, 5, -5, 0, 5])
        ax.axvline(size, color='black', linewidth=0.25)
        ax.axvline(-size, color='black', linewidth=0.25)
        return ax.imshow(
            np.hstack((M[cutoff:-cutoff,cutoff:-cutoff, -1], M[cutoff:-cutoff,cutoff:-cutoff, 1], M[cutoff:-cutoff,cutoff:-cutoff, 0])), cmap=cmap2, vmin=-1, vmax=1, extent=[-3 * size, 3 * size, - size, size],

        )
    if mode == 'chirality' or mode == 'quiver':
        A = cmap((M[:,:,-1] + 1)/4 + 0.5) 
        B = cmap(0.5 - (M[:,:,-1] + 1)/4)
        F = np.sign(M[:,:,-1]) * np.pow(np.abs(M[:,:,-1]), 1/3) * 0.1 + 0.9

        return ax.imshow(A * C[:,:,None] * F[:,:, None] + (1 - C[:,:,None]) * B * F[:,:,None], cmap=cmap, vmin=-1, vmax=1, extent=extent)
    
    if mode == 'spirality':
        A = cmap((M[:,:,-1] + 1)/4 + 0.5) 
        B = cmap(0.5 - (M[:,:,-1] + 1)/4)
        F = np.sign(M[:,:,-1]) * np.pow(np.abs(M[:,:,-1]), 1/3) * 0.1 + 0.9
        base = A * C[:,:,None] * F[:,:, None] + (1 - C[:,:,None]) * B * F[:,:,None]
        E = cmap_chi((M[:,:,-1] + 1)/2) * F[:,:, None]
        

        return ax.imshow(base * (1 - S[:,:,None]) + E * S[:,:,None], cmap=cmap, vmin=-1, vmax=1, extent=extent)
    

    if mode == 'rotation':
        m_z = M[:,:,-1]
        m_z_x = np.gradient(m_z, axis=1)
        m_z_y = np.gradient(m_z, axis=0)
        m_x = M[:,:,0]
        m_y = M[:,:,1]

        sign = np.sign(m_x * m_z_y - m_y * m_z_x)
        cos = (m_x * m_z_x + m_y * m_z_y) ** 2 / ((m_x ** 2 + m_y ** 2) * (m_z_x ** 2 + m_z_y ** 2))
        white = cmap(1)
        blue = cmap(0.75)
        orange = cmap(0.25)

        t = 1 - m_z ** 2

        return ax.imshow(t[:,:,None] * (white * cos[:,:,None] + (1 - cos[:,:,None]) * ( (1 - sign[:,:,None]) / 2 * blue + (sign[:,:,None] + 1) / 2 * orange)) + (1 - t[:,:,None]) * white, cmap=cmap, vmin=0, vmax=1, extent=extent);

def animate_field(data, dir_name: str, show_result: bool = True, mode: Literal['m_z', 'quiver', 'byside', 'chirality']='m_z', cutoff = 50):
    fig, ax = plt.subplots()
    def update(i):
        ax.clear()
        draw_field(ax, data[i], mode=mode, cutoff=cutoff)
        # m = data[i]
        # mx_my = np.linalg.cross(np.gradient(m, axis=0), np.gradient(m, axis=1))
        # v = np.sum(m * mx_my) / (4 * np.pi)
        # ax.text(0,1, f'{round(v, 4)}')

    ani = animation.FuncAnimation(fig, update, repeat=True, frames=len(data), interval=50)
    writer = animation.FFMpegWriter(fps=30)
    ani.save(f"./{dir_name}/{mode}.mp4", writer=writer, dpi=200)

    if show_result:
        os.system(f'code ./{dir_name}/{mode}.mp4')