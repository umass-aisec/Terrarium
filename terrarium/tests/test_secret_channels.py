import asyncio
import unittest
from types import SimpleNamespace

from terrarium.core.blackboard import Megaboard
from terrarium.environments.abstract_environment import AbstractEnvironment
from terrarium.tools.discovery import ToolsetDiscovery


class _StubGraph:
    def number_of_nodes(self):
        return 3

    def number_of_edges(self):
        return 3


class _StubNetwork:
    consolidate_channels_enabled = False

    def __init__(self, groups):
        self._groups = groups
        self.graph = _StubGraph()

    def validate_agents(self, agent_names):
        pass

    def channel_groups(self):
        return self._groups


class _StubProtocol:
    def __init__(self, megaboard):
        self.megaboard = megaboard
        self.calls = []

    async def generate_comm_network(self, participants, context, template=None):
        self.calls.append(
            {
                "participants": list(participants),
                "context": context,
                "template": template,
            }
        )
        blackboard_id = self.megaboard.add_blackboard(list(participants), template=template)
        self.megaboard.post_system_message(blackboard_id, "context", {"message": context})
        return blackboard_id


class _StubEnvironment(AbstractEnvironment):
    def __init__(self, agent_names, groups, env_config):
        self.agent_names = list(agent_names)
        self.env_config = env_config
        self.problem = SimpleNamespace(description="Coordinate on the shared task.")
        self.communication_network = _StubNetwork(groups)
        self.megaboard = Megaboard()
        self.communication_protocol = _StubProtocol(self.megaboard)

    def build_agent_context(self, agent_name, phase, iteration, **kwargs):
        return {}

    def done(self, iteration):
        return True

    def compute_max_joint_reward(self):
        return 0.0

    def joint_reward(self, actions):
        return 0.0

    def agent_reward(self, agent_name, action):
        return 0.0


def _make_env(secret_channels):
    agent_names = ["Alice", "Bob", "Carol"]
    env = _StubEnvironment(
        agent_names, [list(agent_names)], {"secret_channels": secret_channels}
    )
    asyncio.run(env.initialize_communication_network())
    return env


class TestSecretChannelConfig(unittest.TestCase):
    def test_creates_secret_board_with_secret_template_and_registers_it(self):
        env = _make_env([{"members": ["Alice", "Bob"], "context": "whisper seed"}])
        protocol = env.communication_protocol
        self.assertEqual(len(protocol.calls), 2)
        secret_call = protocol.calls[-1]
        self.assertEqual(secret_call["participants"], ["Alice", "Bob"])
        self.assertEqual(secret_call["context"], "whisper seed")
        self.assertEqual(
            secret_call["template"],
            {"private_channel": True, "visibility": "secret"},
        )

        secret_id = env.secret_blackboard_ids[frozenset({"Alice", "Bob"})]
        self.assertEqual(
            env.secret_channel_by_agent, {"Alice": secret_id, "Bob": secret_id}
        )
        board = env.megaboard.blackboards[secret_id]
        self.assertTrue(board.template.get("private_channel"))
        self.assertEqual(board.template.get("visibility"), "secret")

    def test_optional_context_defaults_to_private_channel_seed(self):
        env = _make_env([{"members": ["Alice", "Bob"]}])
        context = env.communication_protocol.calls[-1]["context"]
        self.assertIn("private communication channel", context)
        self.assertIn("Alice", context)
        self.assertIn("Bob", context)

    def test_unknown_member_raises_value_error(self):
        with self.assertRaises(ValueError):
            _make_env([{"members": ["Alice", "Mallory"]}])

    def test_empty_members_raises_value_error(self):
        with self.assertRaises(ValueError):
            _make_env([{"members": []}])

    def test_no_config_initializes_empty_registries(self):
        env = _make_env([])
        self.assertEqual(env.secret_blackboard_ids, {})
        self.assertEqual(env.secret_channel_by_agent, {})
        self.assertEqual(len(env.communication_protocol.calls), 1)


class TestSecretChannelVisibility(unittest.TestCase):
    def setUp(self):
        self.env = _make_env([{"members": ["Alice", "Bob"], "context": "seed"}])
        self.megaboard = self.env.megaboard
        self.secret_id = self.env.secret_blackboard_ids[frozenset({"Alice", "Bob"})]

    def test_member_lists_secret_board_non_member_does_not(self):
        self.assertIn(str(self.secret_id), self.megaboard.get_agent_blackboards("Alice"))
        self.assertIn(str(self.secret_id), self.megaboard.get_agent_blackboards("Bob"))
        self.assertNotIn(str(self.secret_id), self.megaboard.get_agent_blackboards("Carol"))

    def test_get_by_non_member_raises(self):
        with self.assertRaises(ValueError):
            self.megaboard.get(self.secret_id, "Carol")
        self.assertEqual(len(self.megaboard.get(self.secret_id, "Alice")), 1)

    def test_secret_channel_labeled_private_in_prompts(self):
        from terrarium.communication_protocols.sequential import (
            SequentialCommunicationProtocol,
        )

        protocol = SequentialCommunicationProtocol.__new__(
            SequentialCommunicationProtocol
        )
        protocol.megaboard = self.megaboard
        protocol.environment_tools = SimpleNamespace(supports_private_channels=True)
        label = protocol._label_channel(
            str(self.secret_id), self.secret_id, "Alice", "body"
        )
        self.assertEqual(
            label, f"[PRIVATE channel {self.secret_id} \u2014 with Bob]\nbody"
        )


class TestSecretChannelActionScrubbing(unittest.TestCase):
    def test_action_trace_written_to_public_but_not_secret_board(self):
        env = _make_env([{"members": ["Alice", "Bob"], "context": "seed"}])
        megaboard = env.megaboard
        secret_id = env.secret_blackboard_ids[frozenset({"Alice", "Bob"})]
        public_id = env.network_blackboards[frozenset({"Alice", "Bob", "Carol"})]

        megaboard.log_action_to_blackboards(
            "Alice",
            {"action": "schedule_shift", "shift_id": 7},
            {"status": "success"},
            phase="execution",
            iteration=1,
        )

        public_kinds = [e["kind"] for e in megaboard.blackboards[public_id].logs]
        secret_kinds = [e["kind"] for e in megaboard.blackboards[secret_id].logs]
        self.assertIn("action_executed", public_kinds)
        self.assertNotIn("action_executed", secret_kinds)


class TestAllowAgentChannelsOptIn(unittest.TestCase):
    def _discovery(self, tools):
        discovery = ToolsetDiscovery()
        discovery._tools_by_environment["StubEnvironment"] = tools
        return discovery

    def test_allow_agent_channels_true_enables_private_channels(self):
        tools = SimpleNamespace(
            environment=SimpleNamespace(env_config={"allow_agent_channels": True})
        )
        self.assertTrue(self._discovery(tools).supports_private_channels("StubEnvironment"))

    def test_missing_or_false_allow_agent_channels_stays_false(self):
        for env_config in ({}, {"allow_agent_channels": False}):
            tools = SimpleNamespace(environment=SimpleNamespace(env_config=env_config))
            self.assertFalse(
                self._discovery(tools).supports_private_channels("StubEnvironment")
            )

    def test_missing_environment_stays_false(self):
        self.assertFalse(
            self._discovery(SimpleNamespace()).supports_private_channels("StubEnvironment")
        )
        self.assertFalse(
            self._discovery(SimpleNamespace(environment=None)).supports_private_channels(
                "StubEnvironment"
            )
        )

    def test_tools_class_flag_still_enables_private_channels(self):
        tools = SimpleNamespace(supports_private_channels=True, environment=None)
        self.assertTrue(self._discovery(tools).supports_private_channels("StubEnvironment"))


if __name__ == "__main__":
    unittest.main()
