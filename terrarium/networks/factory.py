from __future__ import annotations

from typing import Any, Dict, Sequence

from terrarium.networks.base import CommunicationNetwork
from terrarium.networks.barabasi_albert import BarabasiAlbertNetwork
from terrarium.networks.complete import CompleteNetwork
from terrarium.networks.erdos_renyi import ErdosRenyiNetwork
from terrarium.networks.path import PathNetwork
from terrarium.networks.star import StarNetwork
from terrarium.networks.watts_strogatz import WattsStrogatzNetwork


def build_communication_network(
    agent_names: Sequence[str],
    config: Dict[str, Any],
) -> CommunicationNetwork:
    """Build a CommunicationNetwork from config.

    Supported config keys:
    - `communication_network.topology` (required)
      Values: complete | path | star | erdos_renyi | watts_strogatz | barabasi_albert
    - `communication_network.consolidate_channels` (optional bool; default false)
    - `communication_network.center` (only for star; agent name or integer index)
    - `communication_network.edge_prob` (erdos_renyi; float in [0, 1])
    - `communication_network.k` (watts_strogatz; even int < num_agents)
    - `communication_network.rewire_prob` (watts_strogatz; float in [0, 1])
    - `communication_network.m` (barabasi_albert; int in [1, num_agents-1])
    - random graph seed is taken from `simulation.seed` (for erdos_renyi/ws/ba)
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
        return StarNetwork(
            agents, center=center_name, consolidate_channels=consolidate_channels
        )
    if topology == "random":
        raise ValueError(
            "communication_network.topology='random' is not supported; use 'erdos_renyi'."
        )
    if topology in {"erdos_renyi", "erdos-renyi", "er"}:
        edge_prob = float(cfg["edge_prob"])
        seed_int = int(config["simulation"]["seed"])
        assert isinstance(edge_prob, (int, float)) and 0 <= edge_prob <= 1, (
            f"communication_network.edge_prob must be a float in [0, 1], got: {edge_prob!r}"
        )
        assert isinstance(seed_int, int), (
            f"simulation.seed must be an integer, got: {seed_int!r}"
        )
        return ErdosRenyiNetwork(
            agents,
            edge_prob=edge_prob,
            seed=seed_int,
            consolidate_channels=consolidate_channels,
        )
    if topology in {"watts_strogatz", "watts-strogatz", "ws"}:
        # Parameter aliases: allow both snake_case and more descriptive keys.
        k = cfg.get("k", cfg.get("nearest_neighbors", cfg.get("num_neighbors")))
        p = cfg.get("rewire_prob", cfg.get("rewiring_prob", cfg.get("beta")))
        if k is None or p is None:
            raise ValueError(
                "watts_strogatz requires communication_network.k and communication_network.rewire_prob"
            )
        seed_int = int(config["simulation"]["seed"])
        return WattsStrogatzNetwork(
            agents,
            k=int(k),
            rewire_prob=float(p),
            seed=seed_int,
            consolidate_channels=consolidate_channels,
        )
    if topology in {"barabasi_albert", "barabasi-albert", "ba"}:
        m = cfg.get("m", cfg.get("edges_per_node", cfg.get("num_edges_to_attach")))
        if m is None:
            raise ValueError("barabasi_albert requires communication_network.m")
        seed_int = int(config["simulation"]["seed"])
        return BarabasiAlbertNetwork(
            agents,
            m=int(m),
            seed=seed_int,
            consolidate_channels=consolidate_channels,
        )

    raise ValueError(
        "Unknown communication_network topology: "
        f"{topology!r}. Supported: complete, path, star, erdos_renyi, watts_strogatz, barabasi_albert."
    )
