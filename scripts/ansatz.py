from collections import namedtuple
from dataclasses import dataclass
import scipy.integrate as integrate
import scipy.special as sp
import scipy.interpolate as interp
from .compiled import field
from typing import Callable, List, Literal, NamedTuple, Tuple
import numpy as np
from scipy import optimize as opt # type: ignore

from .problem import Problem # type: ignore

MethodMinimization = Literal[
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov',
    None
]

@dataclass
class Ansatz:
    """
        Arbitrary ansatz
    """
    theta: Callable[[np.ndarray, List[float], Problem], np.ndarray]
    theta_r: Callable[[np.ndarray, List[float], Problem], np.ndarray]

    def energy(self, params, problem: Problem):
        return problem.free_energy(problem.r, self.theta(problem.r, params, problem), self.theta_r(problem.r, params, problem))

    def minimize(self, initial_params: Tuple, problem: Problem, method: MethodMinimization = None) -> opt.OptimizeResult:
        ans: opt.OptimizeResult = opt.minimize(
            lambda params: self.energy(params, problem), 
            initial_params, 
            method=method, 
            tol=problem.tolerance
        )
        return ans


@dataclass
class DomainWallAnsatz:
    """
        Domain wall ansatz
        theta(r) = 2arctan(sinh(R/δ)/sinh(x/δ))
    """

    @staticmethod
    def theta(r, R, delta):
        return 2 * np.arctan(np.sinh(R / delta) / np.sinh(r/ delta))
    
    @staticmethod
    def theta_r(r, R, delta):
        return - 4 * np.sinh(R / delta) * np.cosh(r / delta) / (delta * (np.cosh(2 * r / delta) + 2 * np.sinh(R / delta) ** 2 - 1))

nonskyrmion_theta = interp.Akima1DInterpolator(
        np.linspace(0, 30, 1000),
        np.vectorize(lambda r: integrate.quad(
            lambda xi: xi * field(xi, 1, 0) * (- sp.k1(r) * sp.i0(xi) if xi < r else sp.i1(r) * sp.k0(xi)),
            0,
            np.inf
        )[0])(np.linspace(0, 30, 1000))
    )

@dataclass
class NonskyrmionAnsatz:
    @staticmethod
    def theta(r, alpha):
        return nonskyrmion_theta(r) * (alpha - alpha ** 2 / 37.702 - alpha ** 3 / 70.477)

