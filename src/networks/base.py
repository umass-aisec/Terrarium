from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import networkx as nx


@dataclass(frozen=True, slots=True)
class CommunicationNetwork:
    """A NetworkX-backed communication network.

    Convention used across Terrarium:
    - Nodes are agent names (strings).
    - Edges are potential pairwise connections; blackboard channels are derived
      from this graph via `channel_groups()`.
    """

    # These are set in the constructor of subclasses
    graph: nx.Graph
    consolidate_channels_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.Graph):
            raise TypeError(f"graph must be a networkx.Graph, got: {type(self.graph)!r}")
        if not isinstance(self.consolidate_channels_enabled, bool):
            raise TypeError(
                "consolidate_channels_enabled must be a bool, "
                f"got: {type(self.consolidate_channels_enabled)!r}"
            )

        nodes = list(self.graph.nodes)
        if any(not isinstance(node, str) for node in nodes):
            raise TypeError("Network nodes must be agent name strings.")

        self_loops = [(u, v) for u, v in self.graph.edges if u == v]
        if self_loops:
            raise ValueError(f"Self-loop edges are not supported: {self_loops!r}")

    @property
    def nodes(self) -> List[str]:
        return list(self.graph.nodes)

    @property
    def agent_names(self) -> List[str]:
        return list(self.graph.nodes)

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def validate_agents(self, agent_names: Sequence[str]) -> None:
        """Validate that the network nodes match the given agent names."""
        expected = set(agent_names)
        actual = set(self.graph.nodes)
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "CommunicationNetwork nodes must match environment agent_names. "
                f"Missing={missing!r}, extra={extra!r}."
                f"This is likely an error in the environment logic."
            )

    def consolidate_channels(self) -> List[List[str]]:
        """Return a greedy vertex-disjoint clique cover as communication channels.
        Example: A complete graph with n nodes/agents --> one channel/blackboard with all n agents.

        Algorithm:
        - Repeatedly find a clique (size >= 3), create a multi-participant channel
          for it, remove those nodes, and repeat.
        - When no cliques (size >= 3) remain, create one channel per remaining edge.

        Notes:
        - This is an NP-hard problem, so finding an optimal solution is not feasible.
        - This algorithm may dominate computation time for large graphs.
        - Clique selection is greedy: pick the largest clique each iteration.
        - Channels are vertex-disjoint by construction (nodes are removed after selection).
        """
        working = self.graph.copy()
        channels: List[List[str]] = []

        while True:
            cliques = [sorted(c) for c in nx.find_cliques(working) if len(c) >= 3]
            if not cliques:
                break
            cliques.sort(key=lambda c: (-len(c), c))
            clique = cliques[0]
            channels.append(clique)
            working.remove_nodes_from(clique)

        edges: List[tuple[str, str]] = []
        for u, v in working.edges:
            a, b = sorted((u, v))
            edges.append((a, b))
        for a, b in sorted(edges):
            channels.append([a, b])

        return channels

    def channel_groups(self) -> List[List[str]]:
        """Return communication channel groups (blackboard participants) for this network."""
        if self.consolidate_channels_enabled:
            return self.consolidate_channels()

        edges: List[tuple[str, str]] = []
        for u, v in self.graph.edges:
            a, b = sorted((u, v))
            edges.append((a, b))
        return [[a, b] for a, b in sorted(edges)]

    def save_plot(self, path: str | Path, *, seed: Optional[int] = None) -> Path:
        """Save a PNG visualization of this network graph to the given path."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Plotting requires matplotlib. Install it or disable plotting."
            ) from exc

        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

        import colorsys
        import itertools

        graph = self.graph
        n = graph.number_of_nodes()
        channels = self.channel_groups()

        figsize = (6, 6)
        if n >= 30:
            figsize = (9, 7)
        if n >= 80:
            figsize = (12, 9)

        node_size = 900 if n <= 12 else 600 if n <= 30 else 350
        font_size = 10 if n <= 12 else 8 if n <= 30 else 6
        with_labels = n <= 60

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(
            f"{self.__class__.__name__} "
            f"(nodes={n}, edges={graph.number_of_edges()}, channels={len(channels)})"
        )
        ax.axis("off")

        pos = nx.spring_layout(graph, seed=seed)

        def channel_color(idx: int, total: int, *, is_clique: bool) -> tuple[float, float, float]:
            # Deterministic HSV rainbow palette. Make cliques more saturated so they stand out.
            if total <= 0:
                return (0.2, 0.2, 0.2)
            hue = (idx / total) % 1.0
            saturation = 0.85 if is_clique else 0.6
            value = 0.9 if is_clique else 0.85
            return colorsys.hsv_to_rgb(hue, saturation, value)

        clique_edges: List[tuple[str, str]] = []
        clique_colors: List[tuple[float, float, float]] = []
        pair_edges: List[tuple[str, str]] = []
        pair_colors: List[tuple[float, float, float]] = []
        channel_edges: set[tuple[str, str]] = set()

        for idx, participants in enumerate(channels):
            is_clique = len(participants) >= 3
            color = channel_color(idx, len(channels), is_clique=is_clique)
            participants_sorted = sorted(participants)

            if len(participants_sorted) < 2:
                continue

            if len(participants_sorted) == 2:
                a, b = participants_sorted
                if graph.has_edge(a, b):
                    edge = (a, b)
                    pair_edges.append(edge)
                    pair_colors.append(color)
                    channel_edges.add(edge)
                continue

            for u, v in itertools.combinations(participants_sorted, 2):
                if graph.has_edge(u, v):
                    edge = (u, v)
                    clique_edges.append(edge)
                    clique_colors.append(color)
                    channel_edges.add(edge)

        # Draw remaining edges in light gray so the full topology is still visible.
        remaining_edges = [
            (u, v)
            for u, v in graph.edges
            if tuple(sorted((u, v))) not in channel_edges
        ]
        if remaining_edges:
            nx.draw_networkx_edges(
                graph,
                pos=pos,
                ax=ax,
                edgelist=remaining_edges,
                edge_color="#9ca3af",
                width=1.0,
                alpha=0.25,
            )

        # Draw channel edges on top, colored by channel group.
        if pair_edges:
            nx.draw_networkx_edges(
                graph,
                pos=pos,
                ax=ax,
                edgelist=pair_edges,
                edge_color=pair_colors,
                width=1.8,
                alpha=0.9,
            )
        if clique_edges:
            nx.draw_networkx_edges(
                graph,
                pos=pos,
                ax=ax,
                edgelist=clique_edges,
                edge_color=clique_colors,
                width=2.8,
                alpha=0.95,
            )

        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=node_size, node_color="#93c5fd")
        if with_labels:
            nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=font_size)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
