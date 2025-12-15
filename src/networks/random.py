from __future__ import annotations

import random
from typing import Optional, Sequence

import networkx as nx

from src.networks.base import CommunicationNetwork


class RandomNetwork(CommunicationNetwork):
    """Erdos-Renyi G(n, p) random communication network."""

    def __init__(
        self,
        agent_names: Sequence[str],
        *,
        edge_prob: float,
        seed: Optional[int] = None,
        consolidate_channels: bool = False,
    ) -> None:
        agents = list(agent_names)
        p = float(edge_prob)
        if not 0.0 <= p <= 1.0:
            raise ValueError("communication_network.edge_prob must be in [0.0, 1.0]")

        rng = random.Random(seed)
        g = nx.Graph()
        g.add_nodes_from(agents)
        for i, u in enumerate(agents):
            for v in agents[i + 1 :]:
                if rng.random() < p:
                    g.add_edge(u, v)

        super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)

