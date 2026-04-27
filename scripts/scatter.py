from matplotlib.widgets import Slider 
import pandas as pd #type:ignore
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.interpolate import make_smoothing_spline #type:ignore
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import defaultdict
from numpy.typing import NDArray

from .refined import refined_search
from .compiled import Problem, field, field_r, AnswerShooting
from .ansatz import Ansatz, DomainWallAnsatz
from dataclasses import dataclass


# df = pd.read_csv('notebooks/scripts/fall/fall-DOP853-height=0.3.csv').query("energy < 200 and radius >= 0")

def find_repeating_range(arr, target):
    arr = np.array(arr)
    matches = arr == target
    if not np.any(matches):
        return (-1, -1)
    diff = np.diff(np.concatenate(([False], matches, [False])).astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    lengths = ends - starts
    max_len = np.max(lengths)
    last_max_idx = np.where(lengths == max_len)[0][-1]
    return (int(starts[last_max_idx]), int(ends[last_max_idx]))

def create_spline(alpha, data: pd.DataFrame):
    distances = np.sort(np.unique(data[data['alpha'] == alpha]['distance']))
    radii = []
    flat_distances = []
    
    for distance in distances:
        rs = data.query(f'alpha == {alpha} and distance == {distance}')['radius'].values
        radii.append(
            [
                float(np.min(rs)),
                float(rs[np.argmin(np.abs(rs - np.median(rs)))]),
                float(np.max(rs))
            ] if len(rs) > 1 else [rs[0]]
        )
        if (len(radii[-1]) > 1):
            if (np.min(np.diff(radii[-1])) > 0.08):
                flat_distances.append(distance)
            else: 
                radii.pop()
        else:
            flat_distances.append(distance)
    
    is_multiple = [len(r) > 1 for r in radii]
    repeating_range = find_repeating_range(is_multiple, True)

    if (repeating_range[0] == -1):
        distances_arranged = flat_distances
        radii_arranged = [r[0] for r in radii]
        t = np.linspace(0, 1, len(distances_arranged))
        distances_spline = make_smoothing_spline(t, distances_arranged)
        radii_spline = make_smoothing_spline(t, radii_arranged)
    elif (repeating_range[1] >= len(radii) - 1):
        distances_arranged_1 = np.concatenate([
            flat_distances[:repeating_range[0]],
            flat_distances[repeating_range[0]:repeating_range[1]],
        ])
        radii_arranged_1 = np.concatenate([
            [r[0] for r in radii[:repeating_range[0]]],
            [r[2] for r in radii[repeating_range[0]:repeating_range[1]]],
        ]) # type: ignore 

        distances_arranged_2 = np.concatenate([
            flat_distances[repeating_range[0]:repeating_range[1]][::-1],
            flat_distances[repeating_range[0]:],
        ])
        radii_arranged_2 = np.concatenate([
            [r[1] for r in radii[repeating_range[0]:repeating_range[1]][::-1]],
            [r[0] for r in radii[repeating_range[0]:]]
        ]) # type: ignore 

        t = np.linspace(0, 1, len(distances_arranged_1))
        distances_spline_1 = make_smoothing_spline(t, distances_arranged_1)
        radii_spline_1 = make_smoothing_spline(t, radii_arranged_1)

        t = np.linspace(0, 1, len(distances_arranged_2))
        distances_spline_2 = make_smoothing_spline(t, distances_arranged_2)
        radii_spline_2 = make_smoothing_spline(t, radii_arranged_2)

        def distances_spline(t):
            if (t < 1/2):
                return distances_spline_1(2 * t)
            elif (t == 1/2):
                return np.nan
            else:
                return distances_spline_2(2 * t - 1)

        def radii_spline(t):
            if (t < 1/2):
                return radii_spline_1(2 * t )
            elif (t == 1/2):
                return np.nan
            else:
                return radii_spline_2(2 * t - 1)
        distances_spline = np.vectorize(distances_spline)
        radii_spline = np.vectorize(radii_spline)
    else:
        distances_arranged = np.concatenate([
            flat_distances[:repeating_range[0]],
            flat_distances[repeating_range[0]:repeating_range[1]],
            flat_distances[repeating_range[0]:repeating_range[1]][::-1],
            flat_distances[repeating_range[0]:],
        ])
        radii_arranged = np.concatenate([
            [r[0] for r in radii[:repeating_range[0]]],
            [r[2] for r in radii[repeating_range[0]:repeating_range[1]]],
            [r[1] for r in radii[repeating_range[0]:repeating_range[1]][::-1]],
            [r[0] for r in radii[repeating_range[0]:]]
        ]) # type: ignore 
        t = np.linspace(0, 1, len(distances_arranged))
        distances_spline = make_smoothing_spline(t, distances_arranged)
        radii_spline = make_smoothing_spline(t, radii_arranged)
        
    # for i in range(len(radii)):
    #     if i < repeating_range[0] or i > repeating_range[1]:
    #         plt.scatter(flat_distances[i], radii[i][0], color='black')
    #     else:
    #         plt.scatter(flat_distances[i], radii[i][0], color='red')
    #         plt.scatter(flat_distances[i], radii[i][1], color='green')
    #         plt.scatter(flat_distances[i], radii[i][2], color='blue')
    
    return distances_spline, radii_spline


def update(alpha):
    ax.clear()

    # left_alpha = df.query(f'alpha <= {alpha}').sort_values(by='alpha')['alpha'].values[-1]
    # right_alpha = df.query(f'alpha >= {alpha}').sort_values(by='alpha')['alpha'].values[0]
    left_alpha = alphas[alphas <= alpha][-1]
    right_alpha = alphas[alphas > alpha][0]

    lerp_coef = (alpha - left_alpha) / (right_alpha - left_alpha)
    # distances_spline_left, radii_spline_left = splines[np.argmin(np.abs(alphas - left_alpha))]
    # distances_spline_right, radii_spline_right = splines[np.argmin(np.abs(alphas - right_alpha))]
    
    # t = np.linspace(0, 1, 1001) 
    # distances_spline = distances_spline_left(t) * (1 - lerp_coef) + distances_spline_right(t) * lerp_coef
    # radii_spline = radii_spline_left(t) * (1 - lerp_coef) + radii_spline_right(t) * lerp_coef
    # ax.plot(distances_spline, radii_spline)
    ax.scatter(df[df['alpha'] == left_alpha]['distance'].values, df[df['alpha'] == left_alpha]['energy'].values, s=1, c=(0.2, 0.4, 1, 1 - lerp_coef))
    ax.scatter(df[df['alpha'] == right_alpha]['distance'].values, df[df['alpha'] == right_alpha]['energy'].values, s=1, c=(0.2, 0.4, 1, lerp_coef))
    ax.set_ylim([-50,50])


def separate_branches(x, y, eps=0.1, min_samples=5):
    x = np.array(x)
    y = np.array(y)

    data = np.column_stack((x / x.max(), y))  # normalize x to focus on y clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_

    # Group original indices by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:  # Ignore noise points
            continue
        clusters[label].append(idx)

    # Optional: sort each cluster's indices by x value
    sorted_branches = []
    for indices in clusters.values():
        indices.sort(key=lambda i: x[i])  # Sort by x-value
        sorted_branches.append(indices)

    return sorted_branches

def make_spline(arr, left = None, right = None, lam=None):
    t = np.hstack((
        np.linspace(-1, 0, len(left[0]) + 1)[:-1] if not left is None else [],
        np.linspace(0, 1, len(arr[0])),
        np.linspace(1, 2, len(right[0]) + 1)[1:] if not right is None else [],
    ))
    
    return [
        make_smoothing_spline(t, np.hstack((
            [] if left is None else np.zeros(len(left[0])) if left[i] is None else left[i],
            x,
            [] if right is None else np.zeros(len(right[0])) if right[i] is None else right[i]
        )), lam=lam)
        for i, x in enumerate(arr)  
    ]

t = np.linspace(0, 1, 1000)

class Compilation:
    distances: NDArray
    radii: NDArray
    energies: NDArray
    betas: NDArray
    solutions: list[AnswerShooting]

    def __init__(self, distances = np.array([]), radii = np.array([]), energies = np.array([]), betas=np.array([]), solutions = []):
        self.distances = distances
        self.radii = radii
        self.energies = energies
        self.betas = betas
        self.solutions = [*solutions]

    def add(self, distance, radius, energy, beta = 0, solution: AnswerShooting = None):
        self.distances = np.append(self.distances, distance)
        self.radii = np.append(self.radii, radius)
        self.energies = np.append(self.energies, energy)
        self.betas = np.append(self.betas, beta)
        self.solutions.append(solution)

    def add_solution(self, problem: Problem, solution: AnswerShooting = None):
        self.distances = np.append(self.distances, problem.distance)
        self.radii = np.append(self.radii, problem.answer_R(solution))
        self.energies = np.append(self.energies, problem.answer_energy(solution))
        self.betas = np.append(self.betas, problem.beta)
        self.solutions.append(solution)

    def clear(self):
        self.distances = np.array([])
        self.radii = np.array([])
        self.energies = np.array([])
        self.betas = np.array([])
        self.solutions = []

    def branches(self, eps=0.1, min_samples=5, by = 'radius'):
        if (self.distances.shape[0] == 0):
            return []
        if by == 'radius':
            branches = separate_branches(self.distances, self.radii, eps, min_samples)
        else:
            branches = separate_branches(self.betas, self.radii, eps, min_samples)

        return [Compilation(self.distances[branch], self.radii[branch], self.energies[branch], self.betas[branch]) for branch in branches]

    def filter(self, r=None, beta=None):

        if not (r is None):
            f = (self.radii > r[0]) & (self.radii < r[1])
            if not (beta is None):
                f = f & (self.betas > beta[0]) & (self.betas < beta[1])
        else:
            f = (self.betas > beta[0]) & (self.betas < beta[1])
        return Compilation(self.distances[f], self.radii[f], self.energies[f], self.betas[f], [])
    
    def rotate(self, phi=0):
        return Compilation(self.distances * np.cos(phi) - self.radii * np.sin(phi), self.radii * np.cos(phi) + self.distances * np.sin(phi) , self.energies, self.betas, self.solutions)

distances = np.linspace(0, 12, 300)
eps = 3e-3

def ranged_search(beta, count=10, epsilon=0.3, chi=[1,-1], problem = None, distances=distances, negative=False, both_negative=False):
    chi_buckets = [Compilation() for c in chi]

    if (problem is None):
        problem = Problem(epsilon=epsilon, beta=beta,height = 1, chi=1, method='DOP853')
    else:
        problem.epsilon = epsilon
        problem.beta = beta
        problem.height= 1

    for distance in tqdm(distances):
        problem.distance = distance
        for i, c in enumerate(chi):
            problem.chi = c
            solutions = refined_search(problem, -np.pi/ 2 + eps, np.pi/ 2 - eps, count, 100, exclude_negative=not negative, only_negative=negative and (not both_negative))
            for solution in solutions:
                if (problem.answer_R(solution) > 0.03):
                    chi_buckets[i].add_solution(problem, solution)
    return chi_buckets

def ansatz_search(beta, epsilon=0.3, distances=distances):
    problem = Problem(epsilon=epsilon, beta=beta,height = 1, chi=1, method='DOP853')
    skyrmions = Compilation()
    for distance in tqdm(distances):
        problem.distance = distance
        ansatz = Ansatz(
            lambda r, p, problem: DomainWallAnsatz.theta(r, p[0], p[1]),
            lambda r, p, problem: DomainWallAnsatz.theta_r(r, p[0], p[1])
        )
        res = ansatz.minimize((4, 1), problem, method='Nelder-Mead')
        skyrmions.add(distance, res.x[0], res.fun)
    return skyrmions

def ring_search(beta, epsilon=0.3, count=10, distances=distances):
    rings = Compilation()
    rings.second_radii = np.array([])

    problem = Problem(epsilon=epsilon, beta=beta,height = 1, chi=2, method='DOP853')
    for distance in tqdm(distances):
        problem.distance = distance
        problem.chi = 2
        solutions = refined_search(problem, -np.pi/ 2 + eps, np.pi/ 2 - eps, count, 100, exclude_negative=True)
        for solution in solutions:
            if (problem.answer_R(solution, np.pi * 3 / 2) > 0.03):
                rings.add_solution(problem, solution)
                rings.second_radii = np.append(rings.second_radii, problem.answer_R(solution, np.pi * 3 / 2))
        problem.chi = 3
        solutions = refined_search(problem, -np.pi/ 2 + eps, np.pi/ 2 - eps, count, 100, exclude_negative=True)
        for solution in solutions:
            if (problem.answer_R(solution, np.pi * 5 / 2) > 0.03):
                rings.add_solution(problem, solution)
                rings.second_radii = np.append(rings.second_radii, problem.answer_R(solution, np.pi * 5 / 2))
    return rings

def ring_ansatz_search(beta, epsilon=0.3, distances=distances):
    problem = Problem(epsilon=epsilon, beta=beta,height = 1, chi=1, method='DOP853')
    skyrmions = Compilation()
    for distance in tqdm(distances):
        problem.distance = distance
        ansatz = Ansatz(
            lambda r, p, problem: 2 * DomainWallAnsatz.theta(r, p[0], p[1]),
            lambda r, p, problem: 2 * DomainWallAnsatz.theta_r(r, p[0], p[1])
        )
        res = ansatz.minimize((4, 1), problem, method='Nelder-Mead')
        skyrmions.add(distance, res.x[0], res.fun)
    return skyrmions


class Search:
    ranged = ranged_search
    
    def ranged_beta(betas, count=10, epsilon=0.3, chi=[1,-1], problem = None, distance = 0, negative=False, both_negative=False):
        chi_buckets = [Compilation() for c in chi]

        if (problem is None):
            problem = Problem(epsilon=epsilon, beta=0, distance=distance,height = 1, chi=1, method='DOP853')
        else:
            problem.epsilon = epsilon
            problem.distance = distance
            problem.height= 1

        for beta in tqdm(betas):
            problem.beta = beta
            for i, c in enumerate(chi):
                problem.chi = c
                solutions = refined_search(problem, -np.pi/ 2 + eps, np.pi/ 2 - eps, count, 100, exclude_negative=not negative, only_negative=negative and (not both_negative))
                for solution in solutions:
                    chi_buckets[i].add_solution(problem, solution)
        return chi_buckets
    
    def ring_beta(betas, epsilon=0.3, count=10, distance=0):
        rings = Compilation()
        rings.second_radii = np.array([])

        problem = Problem(epsilon=epsilon, beta=0, distance=distance,height = 1, chi=2, method='DOP853')
        for beta in tqdm(betas):
            problem.beta = beta
            problem.chi = 2
            solutions = refined_search(problem, -np.pi/ 2 + eps, np.pi/ 2 - eps, count, 100, exclude_negative=True)
            for solution in solutions:
                rings.add_solution(problem, solution)
                rings.second_radii = np.append(rings.second_radii, problem.answer_R(solution, np.pi * 3 / 2))
            problem.chi = 3
            solutions = refined_search(problem, -np.pi/ 2 + eps, np.pi/ 2 - eps, count, 100, exclude_negative=True)
            for solution in solutions:
                rings.add_solution(problem, solution)
                rings.second_radii = np.append(rings.second_radii, problem.answer_R(solution, np.pi * 5 / 2))
        return rings