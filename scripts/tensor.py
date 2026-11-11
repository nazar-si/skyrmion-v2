import torch
from torch.func import jacrev

width = 5
height = 5
steps = 800

x = torch.linspace(-width // 2, width // 2, steps)
y = torch.linspace(-width // 2, width // 2, steps)
dx = x[1] - x[0]
dy = y[1] - y[0]

X, Y = torch.meshgrid(x, y, indexing="xy")

def skyrmion_m(X, Y, Xs, Ys, R, delta, eta):
    dx = X - Xs
    dy = Y - Ys

    r = torch.sqrt(dx**2 + dy**2)
    phi = torch.atan2(dy, dx)

    s = torch.sinh(R / delta) / torch.sinh(r / delta)

    theta = 2.0 * torch.atan(s)

    st = torch.sin(theta)
    ct = torch.cos(theta)

    angle = phi + eta

    mx = st * torch.cos(angle)
    my = st * torch.sin(angle)
    mz = ct

    return torch.stack((mx, my, mz), dim=-1)

def electric_field(X, Y, Xt, Yt, h, d):
    dx = X - Xt
    dy = Y - Yt
    r2 = dx**2 + dy**2

    f = (
        -1.0 / (r2 + (h + d)**2)**1.5
        +1.0 / (r2 + (h - d)**2)**1.5
    )

    Ex = dx * f
    Ey = dy * f

    Ez = (
        (h + d) / (r2 + (h + d)**2)**1.5
        +(h - d) / (r2 + (h - d)**2)**1.5
    )

    return torch.stack((Ex, Ey, Ez), dim=-1)

def pearl_B(X, Y, Xv, Yv, l=0.0):
    dx = X - Xv
    dy = Y - Yv

    r = torch.sqrt(dx**2 + dy**2 + l**2)

    Bx = - dx / (r**2)
    By = - dy / (r**2)
    Bz = - 1.0 / r

    return torch.stack((Bx, By, Bz), dim=-1)


def spatial_derivatives(m, dx, dy):
    dx_left = (
        -3.0 * m[:, 0:1, :]
        + 4.0 * m[:, 1:2, :]
        - m[:, 2:3, :]
    ) / (2.0 * dx)

    dx_center = (
        m[:, 2:-1, :] - m[:, 0:-2, :]
    ) / (2.0 * dx)

    dx_right = (
        3.0 * m[:, -1:, :]
        - 4.0 * m[:, -2:-1, :]
        + m[:, -3:-2, :]
    ) / (2.0 * dx)

    dm_dx = torch.cat(
        [dx_left, dx_center, dx_right],
        dim=1,
    )

    dy_bottom = (
        -3.0 * m[0:1, :, :]
        + 4.0 * m[1:2, :, :]
        - m[2:3, :, :]
    ) / (2.0 * dy)

    dy_center = (
        m[2:-1, :, :] - m[0:-2, :, :]
    ) / (2.0 * dy)

    dy_top = (
        3.0 * m[-1:, :, :]
        - 4.0 * m[-2:-1, :, :]
        + m[-3:-2, :, :]
    ) / (2.0 * dy)

    dm_dy = torch.cat(
        [dy_bottom, dy_center, dy_top],
        dim=0,
    )

    return dm_dx, dm_dy

Xt = 0
Yt = 0
epsilon = 0.45
gamma = 0.15
alpha = 0.05
sigma = 0.05
beta = 2
h = 1
d = 0.01

def total_energy(q):
    Xs, Ys, R, delta, eta, Xv, Yv = q

    m = skyrmion_m(X, Y, Xs, Ys, R, delta, eta)

    E = electric_field(
        X, Y,
        Xt, Yt,
        h, d,
    )

    B = pearl_B(
        X, Y,
        Xv, Yv,
        l=0,
    )

    dm_dx, dm_dy = spatial_derivatives(m, dx, dy)
    div_m = dm_dx[..., 0] + dm_dy[..., 1]
    mz = m[..., 2]
    grad_mz_dot_m = (
        m[..., 0] * dm_dx[..., 2]
        + m[..., 1] * dm_dy[..., 2]
    )
    directional_m = (
        m[..., 0, None] * dm_dx
        + m[..., 1, None] * dm_dy
    )
    E_directional_m = (E * directional_m).sum(dim=-1)
    Em = (E * m).sum(dim=-1)


    w_ex = (dm_dx**2).sum(dim=-1) + (dm_dy**2).sum(dim=-1)
    w_an = 1.0 - mz**2
    w_dmi = epsilon * (
        mz * div_m
        - grad_mz_dot_m
    )
    w_E = beta * (
        Em * div_m
        - E_directional_m
        )
    w_B = gamma * (m * B).sum(dim=-1)

    w = w_ex + w_an + w_dmi + w_E + w_B

    return w.sum() * dx * dy

# # Xs, Ys, R, delta, eta, Xv, Yv
# q = torch.tensor([0, 0, 1, 1, 0, 3, 0])

# def magnetisation(q):
#     Xs, Ys, R, delta, eta, Xv, Yv = q

#     return skyrmion_m(
#         X, Y,
#         Xs, Ys,
#         R, delta, eta,
#     )

# # 7 x steps x steps x 3 tensor
# dm_dq = jacrev(magnetisation)(q).permute(3, 0, 1, 2)
# m = magnetisation(q)

# D_density = torch.einsum(
#     "aijc,bijc->abij",
#     dm_dq,
#     dm_dq,
# )

# D = D_density.sum(dim=(-2, -1)) * dx * dy

# # Собираем тензор Леви-Чивиты 
eps3 = torch.zeros(
    (3, 3, 3),
    device=q.device,
    dtype=q.dtype,
)

eps3[0, 1, 2] = 1
eps3[1, 2, 0] = 1
eps3[2, 0, 1] = 1
eps3[0, 2, 1] = -1
eps3[2, 1, 0] = -1
eps3[1, 0, 2] = -1

G_density = torch.einsum(
    "ijc,aijm,bijn,cmn->abij",
    m,
    dm_dq,
    dm_dq,
    eps3,
)

# G = G_density.sum(dim=(-2, -1)) * dx * dy

# A_sk = G + alpha * D

# A = torch.block_diag(
#     A_sk,
#     sigma * torch.eye(
#         2,
#         device=q.device,
#         dtype=q.dtype,
#     ),
# )
# F = total_energy(q)

# q = q.detach().requires_grad_(True)
# gradF, = torch.autograd.grad(F, q)

# qdot = torch.linalg.solve(A, -gradF)

