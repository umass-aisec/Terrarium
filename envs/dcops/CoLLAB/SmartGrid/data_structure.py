from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
import math
import random
from pathlib import Path
import itertools

# --------------------
# Data definitions
# --------------------
@dataclass
class TaskSpec:
    id: str
    consumption: float       # kW per slot
    duration: int            # consecutive slots
    allowed_starts: List[int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HomeSpec:
    id: str
    tasks: List[TaskSpec]

    def to_dict(self) -> dict:
        return {"id": self.id, "tasks": [t.to_dict() for t in self.tasks]}


@dataclass
class InstanceSpec:
    T: int
    S_cap: List[float]
    homes: List[HomeSpec]
    meta: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "T": self.T,
            "S_cap": list(self.S_cap),
            "homes": [h.to_dict() for h in self.homes],
            "meta": self.meta or {},
        }

    @staticmethod
    def from_dict(d: dict) -> "InstanceSpec":
        homes = [HomeSpec(id=h["id"], tasks=[TaskSpec(**t) for t in h["tasks"]]) for h in d["homes"]]
        return InstanceSpec(T=d["T"], S_cap=list(d["S_cap"]), homes=homes, meta=d.get("meta", {}))


# --------------------
# Single global factor
# --------------------
class SustainableCapFactor:
    def __init__(self, T: int, S_cap: List[float], task_index: Dict[Tuple[str, str], TaskSpec]):
        assert len(S_cap) == T
        self.T = T
        self.S_cap = S_cap
        self.task_index = task_index

    def demand_timeseries(self, starts: Dict[Tuple[str, str], int]) -> List[float]:
        D = [0.0] * self.T
        for key, t0 in starts.items():
            task = self.task_index[key]
            for tau in range(t0, min(t0 + task.duration, self.T)):
                D[tau] += task.consumption
        return D

    def main_grid_draw(self, D: List[float]) -> Tuple[List[float], float]:
        G = [max(0.0, D[t] - self.S_cap[t]) for t in range(self.T)]
        return G, float(sum(G))


# --------------------
# Environment wrapper
# --------------------
class NeighborhoodPowerLiteEnv:
    def __init__(self, instance: InstanceSpec):
        self.instance = instance
        self.task_index: Dict[Tuple[str, str], TaskSpec] = {}
        for h in instance.homes:
            for t in h.tasks:
                self.task_index[(h.id, t.id)] = t
        self.factor = SustainableCapFactor(instance.T, instance.S_cap, self.task_index)

    def validate_starts(self, starts: Dict[Tuple[str, str], int]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for key, task in self.task_index.items():
            if key not in starts:
                errors.append(f"Missing start for task {key}")
                continue
            t0 = starts[key]
            if t0 not in task.allowed_starts:
                errors.append(f"Illegal start for task {key}: {t0} not in {task.allowed_starts}")
            if t0 + task.duration > self.instance.T:
                errors.append(f"Task {key} spills past horizon: {t0}+{task.duration} > {self.instance.T}")
        for key in starts:
            if key not in self.task_index:
                errors.append(f"Unknown task key in starts: {key}")
        return (len(errors) == 0), errors

    def evaluate(self, starts: Dict[Tuple[str, str], int]) -> Dict[str, object]:
        ok, errs = self.validate_starts(starts)
        if not ok:
            return {"ok": False, "errors": errs}
        D = self.factor.demand_timeseries(starts)
        G, total_main = self.factor.main_grid_draw(D)
        return {"ok": True, "MainGridEnergy": total_main, "D": D, "G": G}

