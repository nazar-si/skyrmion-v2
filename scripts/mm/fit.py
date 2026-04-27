from typing import Literal

import numpy as np
from scripts.mm.experiment import Experiment
import scipy.optimize as opt

def trajectory(data, experiment: Experiment, method: Literal['center', 'charge'] = 'center'):
    xs = np.array([])
    ys = np.array([])
    W = experiment.width / experiment.dx
    H = experiment.height / experiment.dx
    X, Y = np.meshgrid(np.arange(-W, W) * experiment.dx, -np.arange(-H, H) * experiment.dx)
    for i in range(len(data)):
        if method == 'center':
            weight = (1- data[i,:,:,2]) ** 2
        if method == 'charge':
            weight = np.sum(data[i] * np.linalg.cross(np.gradient(data[i], axis=0), np.gradient(data[i], axis=1)), axis=2)
        norm = np.sum(weight)

        x = np.sum(weight * X) / norm
        y = np.sum(weight * Y) / norm
        xs = np.append(xs, x)
        ys = np.append(ys, y)

    xs = xs - experiment.vortex_to[0]
    ys = ys - experiment.vortex_to[1]
    return xs, ys

def skyrmion(params, X, Y):
    x, y, R, w, eta = params
    r = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    lam = np.sqrt(w ** 2 + 1e-5)
    theta = 2 * np.arctan(np.sinh(R/lam)/np.sinh(r/lam))
    phi = np.arctan2(Y - y, X - x) + eta
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).transpose((1,2,0))

def fit(m, experiment: Experiment, p0=None, return_grid=False, method=None):
    W = experiment.width / experiment.dx
    H = experiment.height / experiment.dx
    X, Y = np.meshgrid(np.arange(-W, W) * experiment.dx, -np.arange(-H, H) * experiment.dx)
    
    weight = (1- m[:,:,2]) ** 2
    norm = np.sum(weight)
    x = np.sum(weight * X) / norm
    y = np.sum(weight * Y) / norm
    res = opt.minimize(lambda params: np.sum(np.abs(skyrmion(params, X, Y) - m)), x0=[x, y,1,1,0] if p0 is None else p0, bounds=[(-W-2,W+2),(-H-2,H+2),(0,None),(0,None),(None, None)], method=method)
    if return_grid:
        return res.x, X, Y
    return res.x