from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx

from src.networks.base import CommunicationNetwork


class BarabasiAlbertNetwork(CommunicationNetwork):
    """Barabasi-Albert preferential attachment communication network."""

    def __init__(
        self,
        agent_names: Sequence[str],
        *,
        m: int,
        seed: Optional[int] = None,
        consolidate_channels: bool = False,
    ) -> None:
        agents = list(agent_names)
        n = len(agents)
        m_int = int(m)

        if n == 0:
            g = nx.Graph()
            super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)
            return

        if n == 1:
            if m_int != 0:
                raise ValueError("communication_network.m must be 0 when num_agents == 1")
            g = nx.Graph()
            g.add_nodes_from(agents)
            super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)
            return

        if m_int < 1:
            raise ValueError("communication_network.m must be >= 1")
        if m_int >= n:
            raise ValueError(f"communication_network.m must be < num_agents (m={m_int}, n={n})")

        g_idx = nx.barabasi_albert_graph(n, m_int, seed=seed)
        mapping = {i: agents[i] for i in range(n)}
        g = nx.relabel_nodes(g_idx, mapping)

        super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)

