from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Any
from pathlib import Path
import json, random, itertools

# --------------- Core datatypes (no style) ---------------

AgentId = str

@dataclass
class Outfit:
    article: str
    color: str
    image: Optional[str] = None  # relative path like "figs/blue_t-shirt.png"

    def as_tuple(self) -> Tuple[str, str]:
        return (self.article, self.color)

@dataclass
class Factor:
    fid: str
    agent_scope: List[AgentId]
    ftype: str
    payload: Dict[str, Any]

    def eval(self, assignment: Dict[AgentId, Outfit]) -> Dict[AgentId, int]:
        # zero points if any variable in scope is unassigned
        if any(a not in assignment for a in self.agent_scope):
            return {a: 0 for a in self.agent_scope}

        # Unary color prefs
        if self.ftype == "PREF_COLOR":
            a = self.agent_scope[0]; want = self.payload["color"]
            return {a: 1 if assignment[a].color == want else 0}
        if self.ftype == "AVOID_COLOR":
            a = self.agent_scope[0]; avoid = self.payload["color"]
            return {a: 1 if assignment[a].color != avoid else 0}

        # Pairwise color match / not-match
        if self.ftype in {"MATCH_COLOR", "NOT_MATCH_COLOR"}:
            a, b = self.agent_scope
            same = assignment[a].color == assignment[b].color
            pts = 1 if (same if self.ftype == "MATCH_COLOR" else not same) else 0
            return {a: pts, b: pts}

        # Future generic hooks (kept for compatibility, but not used here)
        if self.ftype == "ALL_EQUAL_ATTR":
            attr = self.payload["attr"]
            vals = [getattr(assignment[a], attr, None) for a in self.agent_scope]
            ok = all(v == vals[0] for v in vals)
            return {a: (1 if ok else 0) for a in self.agent_scope}
        if self.ftype == "ALL_DIFFERENT_ATTR":
            attr = self.payload["attr"]
            vals = [getattr(assignment[a], attr, None) for a in self.agent_scope]
            ok = len(set(vals)) == len(vals)
            return {a: (1 if ok else 0) for a in self.agent_scope}

        return {a: 0 for a in self.agent_scope}

@dataclass
class FactorGraph:
    agents: List[AgentId]
    factors: List[Factor] = field(default_factory=list)

    def add_factor(self, f: Factor) -> None:
        self.factors.append(f)

    def local_score(self, assignment: Dict[AgentId, Outfit], agent: AgentId) -> int:
        return sum(f.eval(assignment)[agent] for f in self.factors if agent in f.agent_scope)

    def all_local_scores(self, assignment: Dict[AgentId, Outfit]) -> Dict[str, float]:
        return {a: self.local_score(assignment, a) for a in self.agents}

    def global_score(self, assignment: Dict[AgentId, Outfit]) -> int:
        return sum(sum(f.eval(assignment).values()) for f in self.factors)

@dataclass
class Wardrobe:
    options: Dict[AgentId, List[Outfit]]

@dataclass
class PersonalEnvInstance:
    graph: FactorGraph
    wardrobe: Wardrobe
    # prompts intentionally NOT stored here
