from .compiled import Problem
import numpy as np
from numpy.typing import NDArray
import numba as nb

def refined_search(problem: Problem, start: float, end: float, count: int, subcount: int, exclude_negative: bool = False, separate_negative: bool = False, only_negative: bool = False):
    starts, values = problem.angle_range_shooting(problem.r, start, end, count)
    specials = np.argwhere(np.abs(np.diff(values)) > np.abs(values)[:-1] / 2).flatten()
    lefts = np.maximum(specials - 1, 0)
    rights = np.minimum(specials + 2, len(starts + 1))
    signs = np.diff(np.sign(values))
    primitive_zeros = np.argwhere(signs != 0).flatten()
    solutions = []
    negative_solutions = []
    for zero in primitive_zeros:
        if np.all(np.sign(zero - lefts) * np.sign(zero -rights) == 1):
            if (signs[zero] < 0 and exclude_negative):
                continue
            if (signs[zero] > 0 and only_negative):
                continue
            elif (separate_negative and signs[zero] < 0):
                negative_solutions.append(
                    problem.shooting(problem.r, y_prime=np.tan(starts[zero]), upper_boundary=np.tan(starts[zero +1]), lower_boundary=np.tan(starts[zero]), _negative=True)
                )
                continue
            solutions.append(
                problem.shooting(problem.r, y_prime=np.tan(starts[zero]), upper_boundary=np.tan(starts[zero +1]), lower_boundary=np.tan(starts[zero]), _negative=signs[zero] < 0)
            )
    
        
    for idx in specials:
        refined_starts, refined_values = problem.angle_range_shooting(problem.r, starts[max(0, idx - 1)], starts[min(idx + 2, len(starts) -1)], subcount)
        signs = np.diff(np.sign(refined_values))
        zeros = np.argwhere(signs != 0).flatten()
        for zero in zeros:
            if (signs[zero] < 0 and exclude_negative):
                continue
            if (signs[zero] > 0 and only_negative):
                continue
            elif (separate_negative and signs[zero] < 0):
                negative_solutions.append(
                    problem.shooting(problem.r, y_prime=np.tan(refined_starts[zero]), upper_boundary=np.tan(refined_starts[zero +1]), lower_boundary=np.tan(refined_starts[zero]), _negative=True)
                )
                continue
            solutions.append(
                problem.shooting(problem.r, y_prime=np.tan(refined_starts[zero]), upper_boundary=np.tan(refined_starts[zero +1]), lower_boundary=np.tan(refined_starts[zero]), _negative=signs[zero] < 0)
            )
    
    return solutions if not separate_negative else (solutions, negative_solutions)

