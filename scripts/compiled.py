from typing import List, Literal, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator #  type: ignore
from scripts.ellipe import ellipe, ellipe_x, ellipk # type: ignore
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.integrate import solve_bvp, solve_ivp # type: ignore
import pandas as pd # type: ignore
from pathlib import Path
import numba as nb # type: ignore
from numbalsoda import lsoda_sig, lsoda, dop853 #type: ignore

@nb.jit
def pearl_br(r, a):
    """
    Radial component of the azimuthally averaged Pearl vortex field.

    b_r = -Theta(r-a)/r
    """
    if r < a:
        return 0.0

    return -1.0 / r


@nb.jit
def pearl_bz(r, a):
    """
    Out-of-plane component of the azimuthally averaged Pearl vortex field.

    b_z = -2 K(4ar/(a+r)^2) / [pi(a+r)]
    """
    if a == 0:
        # Limiting case: K(0) = pi/2
        return -1.0 / r

    x = 4 * a * r / (a + r) ** 2

    return -2.0 * ellipk(x) / (np.pi * (a + r))

@nb.jit
def magnetic_contribution(r, theta, gamma, distance_vortex):
    br = pearl_br(r, distance_vortex)
    bz = pearl_bz(r, distance_vortex)

    return gamma * (bz * np.sin(theta) - br * np.cos(theta))

@nb.jit
def field(r, h, a):
    if (a == 0):
        return (1 + r ** 2/ h**2)**(-3/2) / h ** 2
    
    d1 = h**2 + (r - a) ** 2
    d2 = h**2 + (r + a) ** 2
    return  (2 * h / np.pi) * ellipe(4 * a * r / d2) / (d1 * np.sqrt(d2))

@nb.jit
def field_a(r, h, a):
    if (a == 0):
        return (1 + r ** 2/ h**2)**(-3/2) / h ** 2
    
    d1 = h**2 + (r - a) ** 2
    d2 = h**2 + (r + a) ** 2
    return (2 * h / np.pi) * ellipe_x(4 * a * r / d2) * (4 * r / d2 - 8 * a * r * (r + a) / d2 ** 2)  / (d1 * np.sqrt(d2)) - field(r, h, a) * (-2 * (r - a) / d1 + (r+a)/d2)

@nb.jit
def field_r(r, h, a):
    if (a == 0):
        return - 3 * r / h ** 4 * (1 + r ** 2 / h ** 2) ** (-5/2)
    
    d1 = h**2 + (r - a) ** 2
    d2 = h**2 + (r + a) ** 2
    return (2 * h / np.pi) * ellipe_x(4 * a * r / d2) * (4 * a / d2 - 8 * a * r * (r + a) / d2 ** 2)  / (d1 * np.sqrt(d2)) - field(r, h, a) * (2 * (r - a) / d1 + (r+a)/d2)

MethodIVP = Literal["DOP853","LSODA"]
MethodShootingValue = Literal['LAST', 'AVG', 'MIN']
BreakOn = Literal['LOWER', 'HIGHER']

@dataclass
class AnswerIVP:
    status: int
    t: NDArray 
    r: NDArray
    y: NDArray
    theta: NDArray
    theta_r: NDArray
    message: str
    success: bool = False

@dataclass
class AnswerShooting:
    r: NDArray
    theta: NDArray
    theta_r: NDArray
    status: int = 0
    message: str = ''
    iterations: int = 0
    success: bool = False

@dataclass
class Interpolation:
    theta: Akima1DInterpolator
    theta_r: Akima1DInterpolator
    start: float = 0
    end: float = 30

@nb.cfunc(lsoda_sig)
def rhs(t, u, du, p):
    """
        p[0] = epsilon
        p[1] = beta
        p[2] = height
        p[3] = distance
        p[4] = gamma
        p[5] = distance_vortex
    """
    inv_t = 1 / t

    # Electric field contribution
    electric_contribution = (
        field(t, p[2], p[3])
        * (1 - np.cos(u[0] * 2))
        * inv_t
        + field_r(t, p[2], p[3])
    )

    # Averaged Pearl vortex magnetic field
    br = pearl_br(t, p[5])
    bz = pearl_bz(t, p[5])

    magnetic_contribution = (
        bz * np.sin(u[0])
        - br * np.cos(u[0])
    )

    du[0] = u[1]

    du[1] = (
        -u[1] * inv_t
        + 0.5 * (1 + inv_t ** 2) * np.sin(u[0] * 2)
        - 2 * p[0] * np.sin(u[0]) ** 2 * inv_t
        - p[1] * electric_contribution
        - p[4] * magnetic_contribution
    )

eq_address = rhs.address # address to ODE function

@nb.jit
def compiled_shooting(r: NDArray, chi: float, epsilon: float, alpha: float, height: float, distance: float, gamma: float, distance_vortex: float, max_iterations=100, y_prime=0, speed=2, upper_boundary = np.inf, lower_boundary = -np.inf, _break_on_up = False, _break_on_down = False, _negative=False, method: MethodIVP = 'LSODA'):
        y_r = y_prime
        lower_bound_found = lower_boundary != -np.inf
        upper_bound_found = upper_boundary != np.inf
        upper_bound = upper_boundary if upper_bound_found else 1e10
        lower_bound = lower_boundary if lower_bound_found else -1e10
        found_solution = np.zeros((2, r.shape[0]))

        for i in range(max_iterations):
            solution = (lsoda(eq_address, np.array([chi * np.pi, y_r]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex])) if method == 'LSODA' else dop853(eq_address, np.array([chi * np.pi, y_r]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex])))[0].T
            value = solution[0][-1]
            if (np.abs(value) < 1e-3 or upper_bound == lower_bound):
                found_solution = solution
                break

            if (value > 0 and not(_negative) or value < 0 and _negative):
                if (_break_on_up):
                    break
                upper_bound = y_r
                upper_bound_found = True
            if (value < 0 and not(_negative) or value > 0 and _negative):
                if (_break_on_down):
                    break
                lower_bound = y_r
                lower_bound_found = True
            
            if (lower_bound_found and upper_bound_found):
                y_r = (lower_bound + upper_bound) / 2
            else:
                y_r -= speed * value
            found_solution = solution 
            

        argmin = np.argmin(np.sum(np.abs(found_solution), axis=0))
        found_solution[0,argmin:] = found_solution[0,argmin] * np.exp(r[:len(r) - argmin] * found_solution[1][argmin])
        found_solution[1,argmin:] = 0

        return found_solution

@nb.jit()
def compiled_probe_shooting(r: NDArray, chi: float, epsilon: float, alpha: float, height: float, distance: float, gamma: float, distance_vortex: float, search_points = 1000, search_intervals=10, refinement_iterations=100, method: MethodIVP = 'LSODA'):
        
        left_boundary = compiled_shooting(r, chi, epsilon, alpha, height, distance, gamma, distance_vortex, max_iterations=300, y_prime=-1e5, speed=5, _break_on_up=True, method=method)
        right_boundary = compiled_shooting(r, chi, epsilon, alpha, height, distance, gamma, distance_vortex, max_iterations=300, y_prime=1e5, speed=5, _break_on_down=True, method=method)
        right_most = right_boundary[1,0]
        left_most = left_boundary[1,0]
        variation = np.abs(right_most - left_most) / 2
        center = (left_most + right_most) / 2
        intervals = np.linspace(center-variation * 1.5, center + variation * 1.5, search_intervals)

        initials = np.zeros((0,))
        ends = np.zeros((0,))
        if (method == 'LSODA'):
            for i in range(1, intervals.shape[0]):
                interval_initials = np.linspace(intervals[i-1], intervals[i], search_points // search_intervals ** 2 )[1:]
                interval_ends = np.array([lsoda(eq_address, np.array([chi * np.pi, initial]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for initial in interval_initials])
                if (np.sign(np.max(interval_ends)) != np.sign(np.min(interval_ends)) or np.var(interval_ends) > np.mean(np.abs(interval_ends))):
                    interval_initials = np.linspace(intervals[i-1], intervals[i], search_points // search_intervals)[1:]
                    interval_ends = np.array([lsoda(eq_address, np.array([chi * np.pi, initial]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for initial in interval_initials])
                initials = np.hstack((initials, interval_initials))
                ends = np.hstack((ends, interval_ends))
        if (method == 'DOP853'):
            for i in range(1, intervals.shape[0]):
                interval_initials = np.linspace(intervals[i-1], intervals[i], search_points // search_intervals ** 2 )[1:]
                interval_ends = np.array([dop853(eq_address, np.array([chi * np.pi, initial]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for initial in interval_initials])
                if (np.sign(np.max(interval_ends)) != np.sign(np.min(interval_ends)) or np.var(interval_ends) > np.mean(np.abs(interval_ends))):
                    interval_initials = np.linspace(intervals[i-1], intervals[i], search_points // search_intervals)[1:]
                    interval_ends = np.array([dop853(eq_address, np.array([chi * np.pi, initial]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for initial in interval_initials])
                initials = np.hstack((initials, interval_initials))
                ends = np.hstack((ends, interval_ends))
        initials = initials[1:]
        ends = ends[1:]
        sign = np.diff(np.sign(ends))
        start_idx = sign != 0
        sign = sign[start_idx]
        lefts = ends[start_idx]
        rights = ends[1:][start_idx[:-1]]
        starts = (lefts + rights) / 2

        answers = np.zeros((starts.shape[0], 2, r.shape[0]), dtype=np.float64)
        
        for i in range(starts.shape[0]):
            solution = compiled_shooting(
                r=r,
                chi=chi,
                epsilon=epsilon,
                alpha=alpha,
                height=height,
                distance=distance,
                gamma=gamma,
                distance_vortex=distance_vortex,
                max_iterations=refinement_iterations,
                y_prime=starts[i], 
                speed=0,
                upper_boundary=lefts[i], 
                lower_boundary=rights[i], 
                _break_on_up=False,
                _break_on_down=False,
                _negative=sign[i] == 2, 
                method=method)
            answers[i] = solution
        return answers

@nb.jit
def range_shooting(r: NDArray, chi: float, epsilon: float, alpha: float, height: float, distance: float, gamma: float, distance_vortex: float, start: float, end: float, count: int, method: MethodIVP = 'LSODA'):
    starts = np.linspace(start, end, count)
    if (method == 'LSODA'):
        return np.array([lsoda(eq_address, np.array([chi * np.pi, start]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for start in starts])
    return np.array([dop853(eq_address, np.array([chi * np.pi, start]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for start in starts])

@nb.jit
def angle_range_shooting(r: NDArray, chi: float, epsilon: float, alpha: float, height: float, distance: float, gamma: float, distance_vortex: float, start: float, end: float, count: int, method: MethodIVP = 'LSODA'):
    starts = np.linspace(start, end, count)
    if (method == 'LSODA'):
        return np.array([lsoda(eq_address, np.array([chi * np.pi, np.tan(start)]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for start in starts])
    return np.array([dop853(eq_address, np.array([chi * np.pi, np.tan(start)]), r, np.array([epsilon, alpha, height, distance, gamma, distance_vortex]))[0][-1,0] for start in starts])


@nb.jit
def smooth_R(r: NDArray, theta: NDArray, chi: float, v: float):
    idx = np.argmin(np.abs(theta - np.sign(chi) * v))
    if ( np.sign(chi) * theta[idx] - v > 0):
        idx += 1
    t = (theta[idx] - np.sign(chi) * v) / (theta[idx] - theta[idx - 1])
    return r[idx - 1] * t + (1-t) * r[idx]

@dataclass
class Problem:
    """Creates skyrmion problem in electric and Pearl vortex magnetic fields"""
    epsilon: float
    beta: float | NDArray | list = 0
    chi: float = 1
    height: float = 1
    helicity: float = 1
    method: MethodIVP = 'LSODA'
    start: float = 0.0005
    end: float = 30
    points: float = 1000
    distance: float | NDArray | list = 0
    gamma: float = 0
    distance_vortex: float = 0
    tolerance: float = 1e-3

    # def second_derivative(self, r: NDArray | float, y: NDArray):
    #     inv_r = 1 / r
    #     field_contribution = field(r, self.height) * (1 - np.cos(y[0] * 2)) * inv_r + field_r(r, self.height)
    #     return - y[1] * inv_r + 0.5 * (1 + 1 * inv_r ** 2) * np.sin(y[0] * 2) - 2 * self.epsilon * np.sin(y[0]) ** 2 * inv_r + self.beta * field_contribution
    def second_derivative(self, r: NDArray | float, y: NDArray):
        inv_r = 1 / r

        # Electric field contribution
        field_contribution = (
            field(r, self.height, self.distance)
            * (1 - np.cos(y[0] * 2))
            * inv_r
            + field_r(r, self.height, self.distance)
        )

        # Pearl vortex magnetic field contribution
        br = pearl_br(r, self.distance_vortex)
        bz = pearl_bz(r, self.distance_vortex)

        magnetic_contribution = (
            bz * np.sin(y[0])
            - br * np.cos(y[0])
        )

        return (
            -y[1] * inv_r
            + 0.5 * (1 + inv_r ** 2) * np.sin(y[0] * 2)
            - 2 * self.epsilon * np.sin(y[0]) ** 2 * inv_r
            + self.beta * field_contribution
            + self.gamma * magnetic_contribution
        )

    def equations(self, r: float | NDArray, y: NDArray) -> NDArray:
        return np.array((
            y[1],
            self.second_derivative(r, y)
        ))
    
    def equations_bvp(self, r: float | NDArray, y: NDArray) -> NDArray:
        return np.vstack((
            y[1],
            self.second_derivative(r, y)
        ))

    @property
    def r(self):
        return np.linspace(self.start, self.end, self.points)

    def bc(self, ya, yb):
        return np.array([ya[0] - self.chi * np.pi, yb[0]])
    
    def guess(self, x: NDArray):
        return np.vstack((
            self.chi * np.pi * np.exp(-x),
            -self.chi * np.pi * np.exp(-x)
        ))

    def solve_bvp(self, mesh: NDArray, verbose: int = 0, max_nodes: int = 10**6, guess: NDArray | None = None):
        """Solve an boundary value problem for a Skyrmion EL."""
        ans = solve_bvp(self.equations_bvp, self.bc, mesh, self.guess(mesh) if guess is None else np.vstack((guess, np.hstack((np.diff(guess) / np.diff(mesh), [0])))), verbose=verbose, max_nodes=max_nodes, tol=self.tolerance)
        return ans
    
    def solve_ivp(self, r: NDArray, y_prime: float):
        if (self.method == "LSODA"):
            solution = lsoda(eq_address, np.array([self.chi * np.pi, y_prime]), r, data=np.array([self.epsilon, self.beta, self.height, self.distance]))[0].T
        else:
            solution = dop853(eq_address, np.array([self.chi * np.pi, y_prime]), r, data=np.array([self.epsilon, self.beta, self.height, self.distance]))[0].T

        return solution[0], solution[1]
    
    def shooting(self, r: NDArray, max_iterations=100, y_prime=0, speed=2, upper_boundary = np.inf, lower_boundary = -np.inf, break_on: BreakOn | None = None, _negative=False) -> AnswerShooting:
        solution = compiled_shooting(r=r, chi=self.chi, epsilon=self.epsilon, alpha=self.beta, height=self.height, distance=self.distance, gamma=self.gamma, distance_vortex=self.distance_vortex, max_iterations=max_iterations, y_prime=y_prime, speed=speed, upper_boundary=upper_boundary, lower_boundary=lower_boundary, _break_on_up=break_on == 'HIGHER', _break_on_down=break_on == 'LOWER', _negative=_negative, method=self.method)

        return AnswerShooting(r, solution[0], solution[1])
    
    def probe_shooting(self, r: NDArray, end_ratio=0.2, search_points = 1000, search_intervals=10, plots=False, refinement_iterations=100, delta=1, convergence_speed=10, convergence_iterations=100) -> List[AnswerShooting]:
        left_boundary = self.shooting(r, convergence_iterations, y_prime=-1e5, speed=convergence_speed, break_on='HIGHER')
        right_boundary = self.shooting(r, convergence_iterations, y_prime=1e5, speed=convergence_speed, break_on="LOWER")

        right_most = right_boundary.theta_r[0]
        left_most = left_boundary.theta_r[0]
        variation = np.abs(right_most - left_most) / 4
        ends = []
        intervals = np.linspace(min(left_most, right_most)-variation, max(left_most, right_most) + variation, search_intervals)

        def get_ends(initials: NDArray):
            ends = []
            for y_prime in initials:
                theta, theta_r = self.solve_ivp(r, y_prime=y_prime)
                ends.append(theta[-1])
            return ends, []
        initials = np.array([])
        for i in range(1, intervals.shape[0]):
            interval_initials = np.linspace(intervals[i-1], intervals[i], search_points // search_intervals ** 2 )[1:]
            interval_ends, interval_answers = get_ends(interval_initials)
            if (np.sign(np.max(interval_ends)) != np.sign(np.min(interval_ends)) or np.var(interval_ends) > np.mean(np.abs(interval_ends))):
                interval_initials = np.linspace(intervals[i-1], intervals[i], search_points // search_intervals)[1:]
                interval_ends, interval_answers = get_ends(interval_initials)
            initials = np.hstack((initials, interval_initials))
            ends += interval_ends
        potential_zeros = np.array([0, 0])
        for i in range(1, len(ends)):
            if (np.sign(ends[i]) != np.sign(ends[i-1])):
                potential_zeros = np.vstack((potential_zeros, (initials[i-1], initials[i])))
        potential_zeros = potential_zeros[1:]
        potential_zeros[0][0] = max(initials[0], potential_zeros[0][0] - delta)
        potential_zeros[-1][-1] = min(initials[-1], potential_zeros[-1][-1] + delta)
        
        for i in range(1, len(potential_zeros)):
            mid = (potential_zeros[i][0] + potential_zeros[i-1][1]) / 2
            potential_zeros[i][0] = max(mid, potential_zeros[i][0] - delta)
            potential_zeros[i-1][1] = min(mid, potential_zeros[i-1][1] + delta)

        if (plots):
            plt.plot(initials, ends)
            plt.scatter(np.mean(potential_zeros, axis=1), [0] * potential_zeros.shape[0], c='r', s=3)
            plt.scatter(potential_zeros[:,0], [0] * potential_zeros.shape[0], c='black', s=3)
            plt.scatter(potential_zeros[:,1], [0] * potential_zeros.shape[0], c='black', s=3)
            
        answers = []
        
        for i, zero in enumerate(potential_zeros):
            ans = self.shooting(r, y_prime=(zero[0] + zero[1]) / 2, lower_boundary=zero[0], upper_boundary=zero[1], _negative=i%2 == 1, max_iterations=refinement_iterations)
            answers.append(ans)
        return answers
    
    def magnetic_energy_density(
        self,
        r: NDArray,
        theta: NDArray,
        spirality = 1
    ):
        br = np.vectorize(pearl_br)(r, self.distance_vortex)
        bz = np.vectorize(pearl_bz)(r, self.distance_vortex)

        return self.gamma * (
            -bz * np.cos(theta)
            -br * np.sin(theta) * spirality
        )

    def field_energy_density(self, r: NDArray, theta: NDArray, theta_r: NDArray):
        if (type(self.distance) == list):
            density = ( 2 * r * theta_r + np.sin(2 * theta))
            s = self.epsilon * density
            for i, d in enumerate(self.distance):
                beta = self.beta if type(self.beta) != list else self.beta[i] 
                s += beta * np.vectorize(field)(r, self.height, d) * density
            return s
        return (self.epsilon + self.beta * np.vectorize(field)(r, self.height, self.distance)) * ( 2 * r * theta_r + np.sin(2 * theta))

    def energy_density(self, r: NDArray, theta: NDArray, theta_r: NDArray):
        field_density = self.field_energy_density(r, theta, theta_r)
        magnetic_density = self.magnetic_energy_density(r, theta)
        return (
            r * theta_r ** 2
            + (1 / r + r) * np.sin(theta) ** 2
            + field_density
            + r * magnetic_density
        )
     
    def field_energy(self, r: NDArray, theta: NDArray, theta_r: NDArray):
        energy_density = self.field_energy_density(r, theta, theta_r)
        return np.pi * np.diff(r) @ (np.copy(energy_density[:-1]) + energy_density[1:])

    def free_energy(self, r: NDArray, theta: NDArray, theta_r: NDArray):
        energy_density = self.energy_density(r, theta, theta_r)
        return np.pi * np.diff(r) @ (np.copy(energy_density[:-1]) + energy_density[1:]) 

    def answer_energy(self, ans: AnswerShooting):
        return self.free_energy(np.array(ans.r), np.array(ans.theta), np.array(ans.theta_r))
    
    def file_name(self, name: str):
        return f'{name}-{self.method}({self.start}:{self.end}:{self.points}).csv'
    def file_path(self):
        return Path(f'./dumps/solutions/height={round(self.height, 5)}/eps={round(float(self.epsilon), 5)}/chi={round(self.chi)}/alpha={round(float(self.beta), 5)}')

    def dump_answer(self, answers: List[AnswerShooting], name: str):
        if (not isinstance(answers, list)):
            answers = [answers]
        dfs: List[pd.DataFrame] = []
        for index, answer in enumerate(answers):
            dfs.append(pd.DataFrame({'n': index, 'r': answer.r, 'theta': answer.theta, 'theta_r': answer.theta_r}))
        df = pd.concat(dfs)
        path = self.file_path()
        path.mkdir(parents=True, exist_ok=True)

        df.to_csv(path / self.file_name(name), index=False)
    
    def load_answer(self, name: str) -> List[AnswerShooting]:
        df = pd.read_csv(self.file_path() / self.file_name(name))
        ns = np.unique(df['n'])
        ans = []
        for n in ns:
            data = df[df['n'] == n]
            ans.append(AnswerShooting(r=data['r'], theta=data['theta'], theta_r=data['theta_r'], status=0, success=True))
        return ans
    
    def range_shooting(self, r: NDArray, start: float, end:float, count: int):
        return np.linspace(np.tan(start), np.tan(end), count), range_shooting(r, self.chi, self.epsilon, self.beta, self.height, self.distance, self.gamma, self.distance_vortex, np.tan(start), np.tan(end), count, self.method)
    def angle_range_shooting(self, r: NDArray, start: float, end:float, count: int):
        return np.linspace(start, end, count), angle_range_shooting(r, self.chi, self.epsilon, self.beta, self.height, self.distance, self.gamma, self.distance_vortex, start, end, count, self.method)
    
    def interpolate(self, ans: AnswerShooting) -> Interpolation:
        theta_interpolated = Akima1DInterpolator(np.concatenate(([0], ans.r)), np.concatenate(([np.pi * self.chi], ans.theta)), extrapolate=True)
        theta_r_interpolated = Akima1DInterpolator(ans.r, ans.theta_r, extrapolate=True)
        
        return Interpolation(theta_interpolated, theta_r_interpolated, 0 if self.start <= self.tolerance else self.start, self.end)
    
    def answer_R(self, ans: AnswerShooting, v: float = np.pi / 2):
        return smooth_R(ans.r, ans.theta, self.chi, v)
    
    def answer_topological_charge(self, ans: AnswerShooting):
        return round(np.sum((ans.theta_r * np.sin(ans.theta))[:-1] * np.diff(ans.r)) / 2, 3)
    
@dataclass
class LinearProblem(Problem):
    chi: float = 0

    @staticmethod
    def from_problem(problem: Problem):
        return LinearProblem(problem.epsilon, problem.beta, 0, problem.height, problem.helicity, problem.method, problem.start, problem.end, problem.points, problem.distance, problem.gamma, problem.distance_vortex, problem.tolerance)

    def second_derivative(self, r: NDArray | float, y: NDArray):
        inv_r = 1 / r
        return - y[1] * inv_r + (1 + 1 * inv_r ** 2) * y[0] + self.beta * field_r(r, self.height)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    problem = Problem(0.5, beta=-0.1, distance=0)
    solutions = problem.probe_shooting(problem.r)
    print(len(solutions))
    ans = solutions[0]
    v = np.linspace(0, np.pi, 10000)
    plt.plot(v, [problem.answer_R(ans, vi) for vi in v])
    plt.plot(ans.theta, ans.r)
    plt.show()