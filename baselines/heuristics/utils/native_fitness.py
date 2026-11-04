"""Optional exact C++ evaluator; the Python implementation remains a fallback."""
import ctypes
import os
from pathlib import Path

import numpy as np

from baselines.heuristics.utils.Config import Config

_library = None
_attempted = False
_cached_instance = None
_geometry = None


def library():
    global _library, _attempted
    if not _attempted:
        _attempted = True
        path = Path(__file__).with_name("_fitness_kernel.so")
        if path.is_file() and os.environ.get("HEURISTIC_PYTHON_ONLY") != "1":
            _library = ctypes.CDLL(str(path))
            array = np.ctypeslib.ndpointer(flags="C_CONTIGUOUS")
            _library.score_routes.argtypes = [ctypes.c_int, ctypes.c_int] + [array] * 8 + [ctypes.c_double] * 5 + [array]
            _library.score_routes.restype = ctypes.c_int
    return _library


def geometry(instance):
    global _cached_instance, _geometry
    if _cached_instance is not instance:
        raw = np.asarray(instance, dtype=np.float64)
        n = len(raw)
        destinations = np.vstack((np.zeros((1, 2)), raw[:, 3:5]))
        sources = np.vstack((np.zeros((1, 2)), raw[:, 1:3]))
        joins = np.abs(destinations[:, None] - destinations[None]).sum(-1)
        pickups = np.abs(destinations[:, None] - sources[None]).sum(-1)
        legs = np.r_[0., np.abs(raw[:, 1:3] - raw[:, 3:5]).sum(-1)]
        due, operation = np.r_[0., raw[:, 5]], np.r_[0., raw[:, 6]]
        _geometry = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in (joins, pickups, legs, due, operation))
        _cached_instance = instance
    return _geometry


def evaluate(instance, sequence_map):
    native = library()
    if native is None:
        return None
    # Current physical formulation starts every vehicle at the origin.
    if any(any(float(x) != 0 for x in pos) for pos in Config.M_position_map.values()):
        return None
    n = len(instance)
    tasks = np.fromiter(sequence_map, dtype=np.int64, count=len(sequence_map))
    mother_pre = np.full(n + 1, -1, dtype=np.int64)
    daughter_pre = np.full(n + 1, -1, dtype=np.int64)
    for task, values in sequence_map.items():
        if task < 1 or task > n:
            return 1e6, 0., 0., False
        m, d = values.get("parent_pre_task"), values.get("child_pre_task")
        if m is None or d is None:
            return 1e6, 0., 0., False
        mother_pre[task], daughter_pre[task] = m, d
    output = np.empty(3, dtype=np.float64)
    feasible = native.score_routes(n, len(tasks), tasks, mother_pre, daughter_pre,
                                  *geometry(instance), float(Config.V), float(Config.T_couple),
                                  float(Config.T_decouple), float(Config.T_load), float(Config.weight), output)
    return (*map(float, output), True) if feasible else (1e6, 0., 0., False)
