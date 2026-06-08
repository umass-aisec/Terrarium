---
title: Networks
---

# Networks

The communication network is the visibility map. It answers a narrow but important question: which agents are allowed to share a blackboard? Terrarium builds the network from agent names and configuration, then the environment turns the resulting channel groups into concrete blackboards.

```python
from terrarium.networks import build_communication_network

agent_names = environment.get_agent_names()
communication_network = build_communication_network(agent_names, config)

environment.set_communication_network(communication_network)
await environment.async_init()
```

## Topologies

Terrarium supports deterministic and random graph families.

| Topology | Family | Use It When |
| --- | --- | --- |
| `complete` | Deterministic | Every agent should be able to talk to every other agent while allowing all communications to be public. |
| `path` | Deterministic | Information should move through a chain. |
| `star` | Deterministic | One central agent should be connected to all others. |
| `erdos_renyi` | Random | You want a random graph controlled by edge probability. |
| `watts_strogatz` | Random | You want local neighborhoods with some rewired long-range links. |
| `barabasi_albert` | Random | You want a hub-heavy graph built by preferential attachment. |

Deterministic families always build the same graph from the same agent list, while random families sample graph structure using `simulation.seed`.

```yaml
communication_network:
  topology: watts_strogatz
  num_agents: 8
  k: 4
  rewire_prob: 0.25
  consolidate_channels: false
```

Random graph families use `simulation.seed`, so repeated runs with the same config build the same topology.

## From Edges to Channels

By default, each edge becomes one two-agent blackboard. That gives you a direct read on how messages move through pairwise links.

```python
channels = communication_network.channel_groups()
# Example: [["agent_0", "agent_1"], ["agent_1", "agent_2"]]
```

When `consolidate_channels` is true, Terrarium tries to combine dense regions into multi-agent blackboards. A complete graph can become one shared channel instead of many pairwise channels. Internally, Terrarium greedily finds cliques with at least three agents, creates one channel for the largest clique it finds, removes those agents from the working graph, and repeats until only pairwise edges remain. Any remaining edges become two-agent channels, and Terrarium adds bridge channels when needed so the resulting channel graph stays connected.

```yaml
communication_network:
  topology: complete
  num_agents: 4
  consolidate_channels: true
```

This is useful for adjusting communication channel complexity which could lead to vastly different multi-agent dynamics and coordination results due to cascading effects across channels.

## Custom Networks

If the built-in topologies are not enough, create a `CommunicationNetwork` from a NetworkX graph. Nodes must be agent names, and the graph must not contain self-loops.

```python
import networkx as nx
from terrarium.networks.base import CommunicationNetwork

graph = nx.Graph()
graph.add_edges_from([
    ("agent_0", "agent_1"),
    ("agent_1", "agent_2"),
    ("agent_2", "agent_3"),
])

network = CommunicationNetwork(graph, consolidate_channels_enabled=False)
network.validate_agents(["agent_0", "agent_1", "agent_2", "agent_3"])
```

Use this when the topology is part of the hypothesis: a mediator, a bottleneck, a coalition, or a partial-information structure that does not fit a standard graph family.

See the [Networks API](../api/networks.md) for generated reference details.
