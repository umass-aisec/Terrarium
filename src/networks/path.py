from __future__ import annotations

from typing import Sequence

import networkx as nx

from src.networks.base import CommunicationNetwork


class PathNetwork(CommunicationNetwork):
    """Path topology where each agent connects to its immediate neighbours."""

    def __init__(self, agent_names: Sequence[str], *, consolidate_channels: bool = False) -> None:
        agents = list(agent_names)
        g = nx.Graph()
        g.add_nodes_from(agents)
        for left, right in zip(agents, agents[1:]):
            g.add_edge(left, right)
        super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)
