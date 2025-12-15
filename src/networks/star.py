from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx

from src.networks.base import CommunicationNetwork


class StarNetwork(CommunicationNetwork):
    """Star topology with a single hub connected to all other agents."""

    def __init__(
        self,
        agent_names: Sequence[str],
        *,
        center: Optional[str] = None,
        consolidate_channels: bool = False,
    ) -> None:
        agents = list(agent_names)
        if not agents:
            raise ValueError("StarNetwork requires at least 1 agent")

        hub = center if center is not None else agents[0]
        if hub not in agents:
            raise ValueError(f"StarNetwork center must be in agent_names, got: {hub!r}")

        g = nx.Graph()
        g.add_nodes_from(agents)
        for node in agents:
            if node == hub:
                continue
            g.add_edge(hub, node)

        super().__init__(graph=g, consolidate_channels_enabled=consolidate_channels)
