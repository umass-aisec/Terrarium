from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
            raise TypeError(
                f"graph must be a networkx.Graph, got: {type(self.graph)!r}"
            )
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
        """Return a greedy clique-based channel cover as communication channels.

        Example: A complete graph with n nodes/agents --> one channel/blackboard with all n agents.

        Algorithm:

        - Repeatedly find a clique (size >= 3), create a multi-participant channel
          for it, remove those nodes, and repeat.
        - When no cliques (size >= 3) remain, create one channel per remaining edge.
        - Finally, ensure the resulting *channel graph* is connected by adding a
          small number of pairwise channels that connect otherwise-disconnected
          channel components (preferring edges that existed in the original graph).

        Notes:

        - This is an NP-hard problem, so finding an optimal solution is not feasible.
        - This algorithm may dominate computation time for large graphs.
        - Clique selection is greedy: pick the largest clique each iteration.
        - Clique channels are vertex-disjoint, but connectivity edges added at the end
          may cause some agents to participate in multiple channels.
        """
        cache_key = "_terrarium_consolidated_channels"
        cached = self.graph.graph.get(cache_key)
        if isinstance(cached, list) and all(isinstance(c, list) for c in cached):
            return cached  # type: ignore[return-value]

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

        channels = self._ensure_channel_connectivity(channels)
        self.graph.graph[cache_key] = channels
        return channels

    def _ensure_channel_connectivity(
        self, channels: List[List[str]]
    ) -> List[List[str]]:
        """
        Ensure the *channel graph* is connected (or as connected as possible) by
        adding a small number of pairwise channels.

        A channel graph has nodes=channels and edges if two channels share a participant.
        """
        # Normalize channels and build quick lookup for existing pair channels.
        normalized: List[List[str]] = []
        pair_edges: Set[Tuple[str, str]] = set()
        for ch in channels:
            if not ch:
                continue
            participants = sorted({str(a) for a in ch})
            if len(participants) < 2:
                # Skip singleton channels (not useful for communication).
                continue
            normalized.append(participants)
            if len(participants) == 2:
                pair_edges.add((participants[0], participants[1]))

        # Track which edges existed before we add any connectivity edges.
        initial_edges: Set[Tuple[str, str]] = {
            tuple(sorted((u, v))) for u, v in self.graph.edges
        }

        # Build channel connectivity via shared participants.
        # Use a union-find for determinism + simplicity.
        parent = list(range(len(normalized)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri = find(i)
            rj = find(j)
            if ri == rj:
                return
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

        agent_to_channel: Dict[str, int] = {}
        for idx, ch in enumerate(normalized):
            for a in ch:
                prev = agent_to_channel.get(a)
                if prev is None:
                    agent_to_channel[a] = idx
                else:
                    union(prev, idx)

        # Component agent sets (union across channels).
        comp_agents: Dict[int, Set[str]] = {}
        for idx, ch in enumerate(normalized):
            root = find(idx)
            comp_agents.setdefault(root, set()).update(ch)

        # Include isolated agents (agents that ended up in no channel).
        present = set().union(*comp_agents.values()) if comp_agents else set()
        for a in self.agent_names:
            if a not in present:
                # Represent each isolated agent as its own component.
                comp_agents.setdefault(-len(comp_agents) - 1, set()).add(str(a))

        if len(comp_agents) <= 1:
            return normalized

        # Deterministic component list: prefer real channel-components (non-negative roots) over
        # singleton-agent components (negative keys), and then by agent name.
        components: List[Set[str]] = []
        for k in sorted(comp_agents.keys(), key=lambda k: (0 if k >= 0 else 1, k)):
            components.append(set(comp_agents[k]))

        def _comp_sort_key(c: Set[str]) -> Tuple[int, str]:
            # Stable ordering for the merge loop.
            first = sorted(c)[0] if c else ""
            return (len(c), first)

        components.sort(key=_comp_sort_key)

        def find_bridge(
            left: Set[str], right: Set[str], *, require_initial: bool
        ) -> Optional[Tuple[str, str]]:
            left_s = sorted(left)
            right_s = sorted(right)
            for u in left_s:
                for v in right_s:
                    a, b = sorted((u, v))
                    if (a, b) in pair_edges:
                        continue
                    if require_initial and (a, b) not in initial_edges:
                        continue
                    return (a, b)
            return None

        # Build a spanning tree across components.
        # Prefer bridges that already exist in the initial graph; if none exist between any
        # remaining components, add a synthetic bridge.
        while len(components) > 1:
            components.sort(key=_comp_sort_key)
            chosen_i = None
            chosen_j = None
            chosen_edge: Optional[Tuple[str, str]] = None

            # First pass: look for any initial-graph bridge between components.
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    edge = find_bridge(
                        components[i], components[j], require_initial=True
                    )
                    if edge is None:
                        continue
                    chosen_i, chosen_j, chosen_edge = i, j, edge
                    break
                if chosen_edge is not None:
                    break

            # Second pass: no initial edges exist between components; create a synthetic bridge.
            if chosen_edge is None:
                chosen_i, chosen_j = 0, 1
                chosen_edge = find_bridge(
                    components[chosen_i], components[chosen_j], require_initial=False
                )
                if chosen_edge is None:  # pragma: no cover
                    # Should be impossible if components are non-empty and disjoint.
                    return normalized

            u, v = chosen_edge
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)
            if (u, v) not in pair_edges:
                normalized.append([u, v])
                pair_edges.add((u, v))

            # Merge the connected components and continue.
            merged = set(components[chosen_i]).union(components[chosen_j])
            # Delete higher index first to keep indices valid.
            hi = max(chosen_i, chosen_j)
            lo = min(chosen_i, chosen_j)
            del components[hi]
            del components[lo]
            components.append(merged)

        return normalized

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

        def channel_color(
            idx: int, total: int, *, is_clique: bool
        ) -> tuple[float, float, float]:
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
            (u, v) for u, v in graph.edges if tuple(sorted((u, v))) not in channel_edges
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

        nx.draw_networkx_nodes(
            graph, pos=pos, ax=ax, node_size=node_size, node_color="#93c5fd"
        )
        if with_labels:
            nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=font_size)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
