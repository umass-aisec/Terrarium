from __future__ import annotations

from typing import Sequence

import networkx as nx

from src.networks.base import CommunicationNetwork


class CompleteNetwork(CommunicationNetwork):
    """Fully-connected (clique) communication network."""

    def __init__(self, agent_names: Sequence[str], *, consolidate_channels: bool = False) -> None:
        super().__init__(
            graph=nx.complete_graph(list(agent_names)),
            consolidate_channels_enabled=consolidate_channels,
        )
