from typing import List, Literal, Tuple
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator #  type: ignore
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.integrate import solve_bvp, solve_ivp # type: ignore
import matplotlib.pyplot as plt
import pandas as pd # type: ignore
from pathlib import Path

def field(r, h):
    return (1 + r ** 2/ h**2)**(-3/2) / h ** 2

def field_r(r, h):
    return - 3 * r / h ** 4 * (1 + r ** 2 / h ** 2) ** (-5/2)

MethodIVP = Literal["RK45","RK23","DOP853","Radau","BDF","LSODA", "MY"]
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

@dataclass
class Problem:
    """Creates skyrmion problem in electric field with defined parameters"""
    epsilon: float
    alpha: float = 0
    chi: float = 1
    height: float = 1
    method: MethodIVP = 'DOP853'
    start: float = 0.0005
    end: float = 30
    points: float = 2000

    """Solve a problem for a Skyrmion EL.

        Parameters
        ----------
        method : string, optional
            Integration method to use:
                * 'MY': My method which is basic, slow, yet most stable.
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
                * 'RK23': Explicit Runge-Kutta method of order 3(2).
                * 'DOP853': Explicit Runge-Kutta method of order 8.
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of order 5.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method based
                on a backward differentiation formula for the derivative
                approximation.
                * 'LSODA': Adams/BDF method with automatic stiffness detection and switching.
        """

    tolerance: float = 1e-3

    def second_derivative(self, r: NDArray | float, y: NDArray):
        inv_r = 1 / r
        field_contribution = field(r, self.height) * (1 - np.cos(y[0] * 2)) * inv_r + field_r(r, self.height)
        return - y[1] * inv_r + 0.5 * (1 + 1 * inv_r ** 2) * np.sin(y[0] * 2) - 2 * self.epsilon * np.sin(y[0]) ** 2 * inv_r + self.alpha * field_contribution

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
    
    def solve_ivp(self, r: NDArray, y_prime: float) -> AnswerIVP:
        if (self.method == 'MY'):
            y = np.zeros((r.shape[0], 2))
            y[0] = [self.chi * np.pi, y_prime]
            for i in range(1, r.shape[0]):
                y[i] = y[i-1] + self.equations(r[i], y[i-1]) * (r[i] - r[i-1])
            y = y.T
            return AnswerIVP(status=0, t=r, r=r, y=y, theta=y[0], theta_r=y[1], message='Success', success=True)
            
        ans: AnswerIVP = solve_ivp(self.equations, [r[0], r[-1]], y0=[self.chi * np.pi, y_prime], t_eval=r, method=self.method, atol=self.tolerance)
        ans.theta = ans.y[0]
        ans.theta_r = ans.y[1]
        ans.r = r
        return ans

    def shooting(self, r: NDArray, max_iterations=100, y_prime=0, speed=2, shooting_method: MethodShootingValue = "MIN", shooting_sample_rate=0.1, shooting_sample_range=0.5, upper_boundary = None, lower_boundary = None, break_on: BreakOn | None = None, _negative=False) -> AnswerShooting:
        y_r = y_prime
        upper_bound = 1e10 if upper_boundary is None else upper_boundary
        lower_bound = -1e10 if lower_boundary is None else lower_boundary
        lower_bound_found = not(lower_boundary is None)
        upper_bound_found = not(upper_boundary is None)
        ans = AnswerShooting(r=np.array(None), theta=np.array(None), theta_r=np.array(None))
        previous_ans = ans
        last_count = int(r.shape[0] * shooting_sample_range)

        for i in range(max_iterations):
            ans = self.solve_ivp(r, y_r) # type: ignore
            if shooting_method == 'LAST':
                value = ans.theta[-1]
            else:
                values = ans.theta[-1:1-last_count:-int(shooting_sample_rate * r.shape[0])]
                if (shooting_method == 'AVG'):
                    value = np.mean(values)
                if (shooting_method == 'MIN'):
                    value = values[np.argmin(np.abs(values)[-last_count:])]
            if (np.abs(value) < self.tolerance):
                ans.iterations = i + 1
                break
            if (i == max_iterations - 1):
                ans.message = 'Shooting iterations limit excised'
                ans.success = False
                ans.status = 1

            if (value > 0 and not(_negative) or value < 0 and _negative):
                if (break_on == 'HIGHER'):
                    ans = previous_ans
                    break
                upper_bound = y_r
                upper_bound_found = True
            if (value < 0 and not(_negative) or value > 0 and _negative):
                if (break_on == 'LOWER'):
                    ans = previous_ans
                    break
                lower_bound = y_r
                lower_bound_found = True
            
            if (lower_bound_found and upper_bound_found):
                y_r = (lower_bound + upper_bound) / 2
            else:
                y_r -= speed * value
            previous_ans = ans

        argmin = np.argmin(np.abs(ans.theta)[r.shape[0] - last_count:]) + r.shape[0] - last_count
        ans.theta[argmin:] = ans.theta[argmin]
        ans.theta_r[argmin:] = 0
        return ans
    
    def probe_shooting(self, r: NDArray, end_ratio=0.2, end_search_method:MethodShootingValue = 'LAST', search_points = 1000, search_intervals=10, plots=False, refinement_iterations=100, use_estimates = False, delta=1) -> List[AnswerShooting]:
        
        left_boundary = self.shooting(r, 100, y_prime=-1e5, speed=10, break_on='HIGHER')
        right_boundary = self.shooting(r, 100, y_prime=1e5, speed=10, break_on="LOWER")

        right_most = right_boundary.theta_r[0]
        left_most = left_boundary.theta_r[0]
        variation = np.abs(right_most - left_most) / 4
        ends = []
        if (use_estimates):
            answers = []
        intervals = np.linspace(min(left_most, right_most)-variation, max(left_most, right_most) + variation, search_intervals)

        def get_ends(initials: NDArray):
            ends = []
            answers = []
            for y_prime in initials:
                ans = self.solve_ivp(r, y_prime=y_prime)
                if (end_search_method == 'LAST'):
                    end = ans.theta[-1]
                else:
                    last_theta = ans.theta[-int(r.shape[0] * end_ratio):]
                    if(end_search_method == "MIN"):
                        end = last_theta[np.argmin(np.abs(last_theta))]
                    else:
                        end = np.mean(last_theta)
                ends.append(end)
                if (use_estimates):
                    answers.append(ans)
            if (use_estimates):
                return ends, answers
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
            if (use_estimates):
                answers += interval_answers
        potential_zeros = np.array([0, 0])
        if (use_estimates):
            potential_answers = []
        for i in range(1, len(ends)):
            if (np.sign(ends[i]) != np.sign(ends[i-1])):
                potential_zeros = np.vstack((potential_zeros, (initials[i-1], initials[i])))
                if (use_estimates):
                    potential_answers.append(answers[i])
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
        if (use_estimates):
            return potential_answers
            
        answers = []
        
        for i, zero in enumerate(potential_zeros):
            ans = self.shooting(r, y_prime=(zero[0] + zero[1]) / 2, lower_boundary=zero[0], upper_boundary=zero[1], _negative=i%2 == 1, shooting_method=end_search_method, max_iterations=refinement_iterations)
            answers.append(ans)
        return answers
    
    def free_energy(self, r: NDArray, theta: NDArray, theta_r: NDArray):
        energy_density = r * theta_r ** 2 + (1 / r + r) * np.sin(theta) ** 2 + (self.epsilon - self.alpha * field(r, self.height)) * ( 2 * r * theta_r + np.sin(2 * theta))
        return np.pi * np.diff(r) @ (np.copy(energy_density[:-1]) + energy_density[1:]) 

    def answer_energy(self, ans: AnswerShooting):
        return self.free_energy(ans.r, ans.theta, ans.theta_r)
    
    def file_name(self, name: str):
        return f'{name}-{self.method}({self.start}:{self.end}:{self.points}).csv'
    def file_path(self):
        return Path(f'./dumps/solutions/height={round(self.height, 5)}/eps={round(float(self.epsilon), 5)}/chi={round(self.chi)}/alpha={round(float(self.alpha), 5)}')

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
    
    def interpolate(self, ans: AnswerShooting) -> Interpolation:
        theta_interpolated = Akima1DInterpolator(np.concatenate(([0], ans.r)), np.concatenate(([np.pi * self.chi], ans.theta)), extrapolate=True)
        theta_r_interpolated = Akima1DInterpolator(ans.r, ans.theta_r, extrapolate=True)
        
        return Interpolation(theta_interpolated, theta_r_interpolated, 0 if self.start <= self.tolerance else self.start, self.end)
        
@dataclass
class LinearProblem(Problem):
    chi: float = 0

    @staticmethod
    def from_problem(problem: Problem):
        return LinearProblem(problem.epsilon, problem.alpha, 0, problem.height, problem.method, problem.start, problem.end, problem.points, problem.tolerance)

    def second_derivative(self, r: NDArray | float, y: NDArray):
        inv_r = 1 / r
        return - y[1] * inv_r + (1 + 1 * inv_r ** 2) * y[0] + self.alpha * field_r(r, self.height)

if __name__ == '__main__':
    pass