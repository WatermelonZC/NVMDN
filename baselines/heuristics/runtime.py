"""Keep completed incumbents available if the wall-clock budget expires."""
callback = None
best_solution = None
last_iteration = 0


class BudgetExpired(BaseException):
    pass


def observe(solution, elapsed, iteration, phase):
    global best_solution, last_iteration
    if best_solution is None or solution.get_fitness() <= best_solution.get_fitness():
        best_solution = solution
    last_iteration = iteration
    if callback is not None:
        callback(solution, elapsed, iteration, phase)
