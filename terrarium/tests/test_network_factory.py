import unittest

from terrarium.networks.barabasi_albert import BarabasiAlbertNetwork
from terrarium.networks.erdos_renyi import ErdosRenyiNetwork
from terrarium.networks.factory import build_communication_network
from terrarium.networks.watts_strogatz import WattsStrogatzNetwork


class CommunicationNetworkFactoryTopologyTests(unittest.TestCase):
    def _config(self, topology: str):
        return {
            "simulation": {"seed": 7},
            "communication_network": {
                "topology": topology,
                "num_agents": 4,
                "edge_prob": 0.5,
                "k": 2,
                "rewire_prob": 0.2,
                "m": 1,
            },
        }

    def test_random_graph_family_canonical_topologies_are_supported(self):
        agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        expected_types = {
            "erdos_renyi": ErdosRenyiNetwork,
            "watts_strogatz": WattsStrogatzNetwork,
            "barabasi_albert": BarabasiAlbertNetwork,
        }

        for topology, expected_type in expected_types.items():
            with self.subTest(topology=topology):
                network = build_communication_network(agents, self._config(topology))
                self.assertIsInstance(network, expected_type)

    def test_random_graph_family_aliases_are_rejected(self):
        agents = ["agent_0", "agent_1", "agent_2", "agent_3"]

        for topology in (
            "random",
            "erdos-renyi",
            "er",
            "watts-strogatz",
            "ws",
            "barabasi-albert",
            "ba",
        ):
            with self.subTest(topology=topology):
                with self.assertRaisesRegex(
                    ValueError, "Unknown communication_network topology"
                ):
                    build_communication_network(agents, self._config(topology))


if __name__ == "__main__":
    unittest.main()
