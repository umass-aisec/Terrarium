from __future__ import annotations

from typing import Any, Dict, Sequence

from src.networks.base import CommunicationNetwork
from src.networks.complete import CompleteNetwork
from src.networks.path import PathNetwork
from src.networks.random import RandomNetwork
from src.networks.star import StarNetwork


def build_communication_network(
    agent_names: Sequence[str],
    config: Dict[str, Any],
) -> CommunicationNetwork:
    """Build a CommunicationNetwork from config.

    Supported config keys:
    - `communication_network.topology` (required)
      Values: complete | path | star | random
    - `communication_network.consolidate_channels` (optional bool; default false)
    - `communication_network.center` (only for star; agent name or integer index)
    - `communication_network.edge_prob` (only for random; float in [0, 1])
    - random graph seed is taken from `simulation.seed`
    """
    cfg = config["communication_network"]
    agents = list(agent_names)
    topology = str(cfg["topology"]).strip().lower()
    consolidate_channels = cfg.get("consolidate_channels", False)
    if not isinstance(consolidate_channels, bool):
        raise TypeError("communication_network.consolidate_channels must be a boolean")

    if topology == "complete":
        return CompleteNetwork(agents, consolidate_channels=consolidate_channels)
    if topology == "path":
        return PathNetwork(agents, consolidate_channels=consolidate_channels)
    if topology == "star":
        center = cfg.get("center")
        if isinstance(center, int):
            center = agents[center]
        center_name = str(center) if center is not None else None
        return StarNetwork(agents, center=center_name, consolidate_channels=consolidate_channels)
    if topology == "random":
        edge_prob = float(cfg["edge_prob"])
        seed_int = int(config["simulation"]["seed"])
        assert isinstance(edge_prob, (int, float)) and 0 <= edge_prob <= 1, f"communication_network.edge_prob must be a float in [0, 1], got: {edge_prob!r}"
        assert isinstance(seed_int, int), f"simulation.seed must be an integer, got: {seed_int!r}"
        return RandomNetwork(
            agents,
            edge_prob=edge_prob,
            seed=seed_int,
            consolidate_channels=consolidate_channels,
        )

    raise ValueError(
        f"Unknown communication_network topology: {topology!r}. Supported: complete, path, star, random."
    )
