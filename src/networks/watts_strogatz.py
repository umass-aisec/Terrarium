from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx

from src.networks.base import CommunicationNetwork


class WattsStrogatzNetwork(CommunicationNetwork):
    """Watts-Strogatz small-world communication network."""

    def __init__(
        self,
        agent_names: Sequence[str],
        *,
        k: int,
        rewire_prob: float,
        seed: Optional[int] = None,
        consolidate_channels: bool = False,
    ) -> None:
        agents = list(agent_names)
        n = len(agents)

        k_int = int(k)
        p = float(rewire_prob)
        if n == 0:
            g = nx.Graph()
            super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)
            return

        if k_int < 0:
            raise ValueError("communication_network.k must be >= 0")
        if k_int >= n:
            raise ValueError(f"communication_network.k must be < num_agents (k={k_int}, n={n})")
        if k_int % 2 != 0:
            raise ValueError(f"communication_network.k must be even for Watts-Strogatz (k={k_int})")
        if not 0.0 <= p <= 1.0:
            raise ValueError("communication_network.rewire_prob must be in [0.0, 1.0]")

        g_idx = nx.watts_strogatz_graph(n, k_int, p, seed=seed)
        mapping = {i: agents[i] for i in range(n)}
        g = nx.relabel_nodes(g_idx, mapping)

        super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)

