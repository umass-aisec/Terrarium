import unittest

from terrarium.core.blackboard import Megaboard
from terrarium.tools.discovery import ToolsetDiscovery


class TestAgentCreatedChannels(unittest.TestCase):
    def setUp(self):
        self.board = Megaboard()
        self.public = self.board.add_blackboard(["A", "B", "C"])

    def _create(self, agent, **arguments):
        return self.board.handle_tool_call("create_channel", agent, arguments)

    def test_creates_private_channel_visible_only_to_participants(self):
        result = self._create("A", agent_ids=["B"], message="want to swap?")
        channel_id = result["blackboard_id"]

        self.assertEqual(result["participants"], ["A", "B"])
        self.assertTrue(self.board.is_private_channel(channel_id))
        self.assertNotEqual(channel_id, self.public)
        self.assertIn(str(channel_id), self.board.get_agent_blackboards("B"))
        self.assertNotIn(str(channel_id), self.board.get_agent_blackboards("C"))
        with self.assertRaises(ValueError):
            self.board.get(channel_id, "C")

        contents = [
            e["payload"].get("content") for e in self.board.get(channel_id, "A")
        ]
        self.assertIn("want to swap?", contents)

    def test_reuses_channel_with_same_membership(self):
        first = self._create("A", agent_ids=["B"])["blackboard_id"]
        second = self._create("B", agent_ids=["A"])
        self.assertEqual(second["blackboard_id"], first)
        self.assertFalse(second["created"])

    def test_rejects_unreachable_and_empty_invites(self):
        self.assertIn("error", self._create("A", agent_ids=["Stranger"]))
        self.assertIn("error", self._create("A", agent_ids=["A"]))
        self.assertIn("error", self._create("A", agent_ids=[]))

    def test_post_message_defaults_to_the_agents_own_channel(self):
        private = self._create("A", agent_ids=["B"])["blackboard_id"]
        solo = self.board.add_blackboard(["D", "E"])

        # An agent outside blackboard 0 posts to its own channel, not to 0.
        self.board.handle_tool_call("post_message", "D", {"message": "hi"})
        self.assertEqual(len(self.board.get(solo, "D")), 1)
        self.assertEqual(len(self.board.get(self.public, "A")), 0)

        # An agent in several channels defaults to its first (public) one.
        self.board.handle_tool_call("post_message", "A", {"message": "hello all"})
        self.assertEqual(len(self.board.get(self.public, "A")), 1)
        self.assertEqual(
            [e["kind"] for e in self.board.get(private, "A")], ["context"]
        )

    def test_tool_exposure_is_opt_in_per_environment(self):
        discovery = ToolsetDiscovery()
        seat_tools = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard(
                "planning", "IsThisSeatTakenEnvironment"
            )
        }
        other_tools = {
            t["function"]["name"]
            for t in discovery.get_tools_for_blackboard(
                "planning", "MeetingSchedulingEnvironment"
            )
        }

        self.assertIn("create_channel", seat_tools)
        self.assertIn(
            "create_channel",
            discovery.get_blackboard_tool_names("IsThisSeatTakenEnvironment"),
        )
        self.assertNotIn("create_channel", other_tools)
        self.assertNotIn(
            "create_channel",
            discovery.get_blackboard_tool_names("MeetingSchedulingEnvironment"),
        )


if __name__ == "__main__":
    unittest.main()
