"""
Tests for blackboard compaction, using a fake model client.

These pin down what compact_events does today: when it compacts, how it builds
the prompt text, what each mechanism sends to the summarizer, and what it logs.
No real model is called.
"""

import itertools
import unittest

from terrarium.compaction.compactor import (
    DEFAULT_EVICT_KINDS,
    MECHANISMS,
    _BULLET_RULES,
    _MAX_SUMMARY_TOKENS,
    _count_tokens,
    compact_events,
)
from terrarium.core.blackboard import format_blackboard_events_for_prompt

MODEL = "fake-model"


class FakeClient:
    """Stands in for an LLM client: records each request and returns a canned summary."""

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def init_context(self, system_prompt, user_prompt):
        return {"system": system_prompt, "user": user_prompt}

    def generate_response(self, input, params):
        self.calls.append({"system": input["system"], "user": input["user"], "params": dict(params)})
        reply = self.reply if self.reply is not None else f"- summary {len(self.calls)}"
        return None, reply


class RecordingLogger:
    def __init__(self):
        self.entries = []

    def log_compaction(self, **kwargs):
        self.entries.append(kwargs)


class BrokenLogger:
    def log_compaction(self, **kwargs):
        raise RuntimeError("disk full")


_ids = itertools.count()


def msg(agent, content):
    """A chat message shaped like Megaboard.post output. Keep digits out of content
    unless the test is about the extractive mechanism, which treats digits as signal."""
    return {"id": f"e{next(_ids)}", "agent": agent, "kind": "communication",
            "payload": {"content": content, "phase": "planning", "iteration": 1}}


def context(message):
    return {"id": f"e{next(_ids)}", "agent": "SYSTEM", "kind": "context", "payload": {"message": message}}


def action(agent, marker):
    return {"id": f"e{next(_ids)}", "agent": agent, "kind": "action_executed",
            "payload": {"action_type": "move", "details": {"to_seat": marker}}}


def chat(*contents):
    return [msg("alice" if i % 2 == 0 else "bob", c) for i, c in enumerate(contents)]


def run(events, client=None, **kwargs):
    """compact_events with a fake client and a threshold of 0, so compaction triggers
    whenever there is anything old enough to summarize."""
    kwargs.setdefault("token_threshold", 0)
    kwargs.setdefault("keep_recent", 2)
    return compact_events(events, llm_client=client, model_name=MODEL, **kwargs)


class WhenCompactionRuns(unittest.TestCase):
    def test_below_threshold_returns_raw_transcript_without_calling_model(self):
        events = chat("old alpha", "old bravo", "new charlie")
        client = FakeClient()
        text = format_blackboard_events_for_prompt(events)
        out = run(events, client, token_threshold=_count_tokens(text) + 100)
        self.assertEqual(out, text)
        self.assertEqual(client.calls, [])

    def test_threshold_is_inclusive(self):
        events = chat("old alpha", "old bravo", "new charlie")
        tokens = _count_tokens(format_blackboard_events_for_prompt(events))
        at, over = FakeClient(), FakeClient()
        run(events, at, token_threshold=tokens)
        run(events, over, token_threshold=tokens - 1)
        self.assertEqual(len(at.calls), 0, "count equal to threshold must not compact")
        self.assertEqual(len(over.calls), 1, "count above threshold must compact")

    def test_tokens_estimated_as_characters_over_four(self):
        self.assertEqual(_count_tokens("x" * 40), 10)
        self.assertEqual(_count_tokens("x" * 43), 10)

    def test_no_client_or_no_model_never_compacts(self):
        # enough old events that compaction would otherwise run
        events = chat("old alpha", "old bravo", "old echo", "old foxtrot", "new charlie")
        text = format_blackboard_events_for_prompt(events)
        self.assertEqual(
            compact_events(events, llm_client=None, model_name=MODEL, token_threshold=0, keep_recent=1), text)
        client = FakeClient()
        self.assertEqual(
            compact_events(events, llm_client=client, model_name=None, token_threshold=0, keep_recent=1), text)
        self.assertEqual(client.calls, [])

    def test_nothing_older_than_keep_recent_returns_raw(self):
        events = chat("alpha", "bravo")
        client = FakeClient()
        out = run(events, client, keep_recent=2)
        self.assertEqual(out, format_blackboard_events_for_prompt(events))
        self.assertEqual(client.calls, [])

    def test_unknown_mechanism_raises(self):
        with self.assertRaises(ValueError):
            run(chat("alpha"), FakeClient(), mechanism="hierarchical")

    def test_non_list_events_treated_as_empty(self):
        empty = format_blackboard_events_for_prompt([])
        self.assertEqual(run(None, FakeClient()), empty)
        self.assertEqual(run({"kind": "communication"}, FakeClient()), empty)


class PromptAssembly(unittest.TestCase):
    def test_baseline_sections_and_what_goes_where(self):
        events = chat("old alpha", "old bravo", "new charlie", "new delta")
        client = FakeClient(reply="- they agreed")
        out = run(events, client, keep_recent=2)

        self.assertEqual(len(client.calls), 1)
        sent = client.calls[0]["user"]
        self.assertTrue(sent.startswith(_BULLET_RULES))
        self.assertIn("old alpha", sent)
        self.assertIn("old bravo", sent)
        self.assertNotIn("new charlie", sent, "recent events must not be summarized")

        self.assertEqual(
            out,
            "[Summary of earlier conversation]\n- they agreed\n\n"
            "[Recent messages]\n" + format_blackboard_events_for_prompt(events[-2:]),
        )
        self.assertNotIn("old alpha", out, "summarized events must not appear verbatim")

    def test_keep_recent_zero_summarizes_everything(self):
        events = chat("alpha", "bravo")
        client = FakeClient(reply="- s")
        out = run(events, client, keep_recent=0)
        self.assertIn("alpha", client.calls[0]["user"])
        self.assertIn("bravo", client.calls[0]["user"])
        self.assertTrue(out.endswith("[Recent messages]\n" + format_blackboard_events_for_prompt([])))

    def test_every_mechanism_produces_summary_and_recent_sections(self):
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism):
                events = chat("old alpha", "old bravo", "new charlie")
                out = run(events, FakeClient(reply="- s"), mechanism=mechanism, keep_recent=1, cache={})
                self.assertIn("[Summary of earlier conversation]\n", out)
                self.assertTrue(out.endswith("[Recent messages]\n" + format_blackboard_events_for_prompt(events[-1:])))


class Pinning(unittest.TestCase):
    def setUp(self):
        self.rules = context("Channel rules: be polite")
        self.events = [self.rules] + chat("old alpha", "old bravo", "new charlie")

    def test_pinned_context_is_kept_verbatim_and_not_summarized(self):
        client = FakeClient(reply="- s")
        out = run(self.events, client, pin_context_events=True, keep_recent=1)
        self.assertNotIn("Channel rules", client.calls[0]["user"])
        self.assertTrue(out.startswith(
            "[Standing context]\n" + format_blackboard_events_for_prompt([self.rules]) + "\n\n"
        ))

    def test_without_pinning_context_is_summarized_like_anything_else(self):
        client = FakeClient(reply="- s")
        out = run(self.events, client, pin_context_events=False, keep_recent=1)
        self.assertIn("Channel rules", client.calls[0]["user"])
        self.assertNotIn("[Standing context]", out)

    def test_pinned_events_do_not_count_toward_the_threshold(self):
        long_rules = context("Channel rules: " + "be polite " * 200)
        events = [long_rules] + chat("old alpha", "old bravo", "new charlie")
        chat_only = _count_tokens(format_blackboard_events_for_prompt(events[1:]))
        pinned, unpinned = FakeClient(), FakeClient()
        run(events, pinned, pin_context_events=True, keep_recent=1, token_threshold=chat_only)
        run(events, unpinned, pin_context_events=False, keep_recent=1, token_threshold=chat_only)
        self.assertEqual(len(pinned.calls), 0)
        self.assertEqual(len(unpinned.calls), 1)

    def test_below_threshold_keeps_context_in_natural_order(self):
        out = run(self.events, FakeClient(), pin_context_events=True, token_threshold=10_000)
        self.assertEqual(out, format_blackboard_events_for_prompt(self.events))


class Anchored(unittest.TestCase):
    def test_first_call_summarizes_and_fills_cache(self):
        cache, client = {}, FakeClient(reply="- first")
        events = chat("old alpha", "old bravo", "new charlie")
        run(events, client, mechanism="anchored", keep_recent=1, cache=cache)
        self.assertEqual(cache, {"summary": "- first", "summarized_count": 2})
        self.assertTrue(client.calls[0]["user"].startswith(_BULLET_RULES))

    def test_later_call_folds_in_only_new_events(self):
        cache = {}
        events = chat("old alpha", "old bravo", "new charlie")
        run(events, FakeClient(reply="- first"), mechanism="anchored", keep_recent=1, cache=cache)

        events = events + chat("later echo")
        client = FakeClient(reply="- second")
        run(events, client, mechanism="anchored", keep_recent=1, cache=cache)
        sent = client.calls[0]["user"]
        self.assertIn("EXISTING SUMMARY:\n- first", sent)
        self.assertIn("new charlie", sent, "the event that just aged out should be folded in")
        self.assertNotIn("old alpha", sent, "already-summarized events must not be resent")
        self.assertEqual(cache, {"summary": "- second", "summarized_count": 3})

    def test_no_new_events_reuses_cached_summary_without_calling_model(self):
        cache = {}
        events = chat("old alpha", "old bravo", "new charlie")
        run(events, FakeClient(reply="- first"), mechanism="anchored", keep_recent=1, cache=cache)
        client = FakeClient()
        out = run(events, client, mechanism="anchored", keep_recent=1, cache=cache)
        self.assertEqual(client.calls, [])
        self.assertIn("[Summary of earlier conversation]\n- first", out)

    def test_without_cache_summarizes_from_scratch_every_time(self):
        events = chat("old alpha", "old bravo", "new charlie")
        client = FakeClient()
        run(events, client, mechanism="anchored", keep_recent=1, cache=None)
        run(events, client, mechanism="anchored", keep_recent=1, cache=None)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(c["user"].startswith(_BULLET_RULES) for c in client.calls))

    def test_shrunk_history_starts_over(self):
        cache = {"summary": "- stale", "summarized_count": 10}
        client = FakeClient(reply="- fresh")
        run(chat("old alpha", "old bravo", "new charlie"), client, mechanism="anchored", keep_recent=1, cache=cache)
        self.assertNotIn("EXISTING SUMMARY", client.calls[0]["user"])
        self.assertEqual(cache, {"summary": "- fresh", "summarized_count": 2})


class Eviction(unittest.TestCase):
    def test_default_drops_action_events_before_summarizing(self):
        self.assertEqual(DEFAULT_EVICT_KINDS, frozenset({"action_executed"}))
        events = [msg("alice", "old alpha"), action("bob", "seat_zulu"), msg("bob", "new charlie")]
        client = FakeClient()
        run(events, client, mechanism="eviction", keep_recent=1)
        self.assertIn("old alpha", client.calls[0]["user"])
        self.assertNotIn("seat_zulu", client.calls[0]["user"])

    def test_custom_evict_kinds(self):
        events = [msg("alice", "old alpha"), action("bob", "seat_zulu"), msg("bob", "new charlie")]
        client = FakeClient()
        run(events, client, mechanism="eviction", keep_recent=1, evict_kinds={"communication"})
        self.assertNotIn("old alpha", client.calls[0]["user"])
        self.assertIn("seat_zulu", client.calls[0]["user"])


class Extractive(unittest.TestCase):
    def test_high_signal_kept_verbatim_rest_summarized(self):
        events = [msg("alice", "I will take the aisle"), msg("bob", "nice weather"),
                  msg("alice", "row 3 is loud"), msg("bob", "new charlie")]
        client = FakeClient(reply="- s")
        out = run(events, client, mechanism="extractive", keep_recent=1)

        sent = client.calls[0]["user"]
        self.assertIn("nice weather", sent)
        self.assertNotIn("I will take the aisle", sent)
        self.assertNotIn("row 3 is loud", sent)
        self.assertIn(
            "[Notable events kept verbatim]\n" + format_blackboard_events_for_prompt([events[0], events[2]]),
            out,
        )

    def test_markers_match_case_insensitively(self):
        events = [msg("alice", "WE AGREE"), msg("bob", "fine"), msg("bob", "new charlie")]
        client = FakeClient()
        run(events, client, mechanism="extractive", keep_recent=1)
        self.assertNotIn("WE AGREE", client.calls[0]["user"])

    def test_extract_limit_keeps_oldest_and_summarizes_overflow(self):
        events = [msg("alice", "deal one"), msg("bob", "deal two"), msg("alice", "deal three"),
                  msg("bob", "new charlie")]
        client = FakeClient()
        out = run(events, client, mechanism="extractive", keep_recent=1, extract_limit=2)
        self.assertIn("deal three", client.calls[0]["user"])
        self.assertIn("deal one", out)
        self.assertIn("deal two", out)

    def test_all_old_events_high_signal_makes_no_model_call(self):
        events = [msg("alice", "deal one"), msg("bob", "new charlie")]
        client = FakeClient()
        out = run(events, client, mechanism="extractive", keep_recent=1)
        self.assertEqual(client.calls, [])
        self.assertIn("[Summary of earlier conversation]\nNo decisions made.", out)

    def test_events_without_message_content_are_never_extracted(self):
        events = [action("bob", "seat_9_9"), msg("alice", "hello"), msg("bob", "new charlie")]
        client = FakeClient()
        out = run(events, client, mechanism="extractive", keep_recent=1)
        self.assertIn("seat_9_9", client.calls[0]["user"])
        self.assertNotIn("[Notable events kept verbatim]", out)


class ReaderAndStructure(unittest.TestCase):
    def test_query_conditioned_names_the_reader_and_phase(self):
        client = FakeClient()
        run(chat("old alpha", "new charlie"), client, mechanism="query_conditioned", keep_recent=1,
            agent_name="agent_2", phase="execution")
        sent = client.calls[0]["user"]
        self.assertTrue(sent.startswith("This summary will be read by agent_2, who is about to act during the execution phase."))
        self.assertIn(_BULLET_RULES, sent)

    def test_query_conditioned_without_reader_matches_baseline_prompt(self):
        conditioned, baseline = FakeClient(), FakeClient()
        events = chat("old alpha", "new charlie")
        run(events, conditioned, mechanism="query_conditioned", keep_recent=1)
        run(events, baseline, mechanism="baseline", keep_recent=1)
        self.assertEqual(conditioned.calls[0]["user"], baseline.calls[0]["user"])

    def test_structured_asks_for_labeled_sections(self):
        client = FakeClient()
        run(chat("old alpha", "new charlie"), client, mechanism="structured", keep_recent=1)
        for label in ("DECISIONS:", "COMMITMENTS:", "OPEN REQUESTS:", "STATE FACTS:"):
            self.assertIn(label, client.calls[0]["user"])


def token_keys(limit):
    """The token limit under every key the clients read."""
    return {"max_completion_tokens": limit, "max_output_tokens": limit, "max_tokens": limit}


class SummaryRequest(unittest.TestCase):
    def test_request_defaults_to_fixed_token_cap(self):
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism):
                client = FakeClient()
                run(chat("old alpha", "new charlie"), client, mechanism=mechanism, keep_recent=1, cache={})
                self.assertEqual(client.calls[0]["params"], {"model": MODEL, **token_keys(_MAX_SUMMARY_TOKENS)})

    def test_request_uses_configured_params(self):
        params = {"max_tokens": 128, "temperature": 0.0}
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism):
                client = FakeClient()
                run(chat("old alpha", "new charlie"), client, mechanism=mechanism, keep_recent=1, cache={}, params=params)
                self.assertEqual(client.calls[0]["params"], {"model": MODEL, "temperature": 0.0, **token_keys(128)})

    def test_anchored_update_uses_configured_params(self):
        client, cache = FakeClient(), {}
        params = {"max_tokens": 128}
        run(chat("old alpha", "new charlie"), client, mechanism="anchored", keep_recent=1, cache=cache, params=params)
        run(chat("old alpha", "new charlie", "newer delta"), client, mechanism="anchored", keep_recent=1, cache=cache, params=params)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("EXISTING SUMMARY", client.calls[1]["user"])
        self.assertEqual(client.calls[1]["params"], {"model": MODEL, **token_keys(128)})

    def test_configured_params_cannot_change_model_or_send_none(self):
        client = FakeClient()
        run(chat("old alpha", "new charlie"), client, keep_recent=1, params={"model": "other", "temperature": None})
        self.assertEqual(client.calls[0]["params"], {"model": MODEL, **token_keys(_MAX_SUMMARY_TOKENS)})


class Logging(unittest.TestCase):
    def test_skipped_call_is_logged_with_identical_before_and_after(self):
        log = RecordingLogger()
        events = chat("alpha", "bravo")
        run(events, FakeClient(), compaction_logger=log, token_threshold=10_000,
            agent_name="agent_1", blackboard_id=0, phase="planning", iteration=3)
        (entry,) = log.entries
        text = format_blackboard_events_for_prompt(events)
        self.assertFalse(entry["triggered"])
        self.assertEqual(entry["pre_text"], text)
        self.assertEqual(entry["post_text"], text)
        self.assertEqual((entry["agent_name"], entry["blackboard_id"], entry["phase"], entry["iteration"]),
                         ("agent_1", 0, "planning", 3))

    def test_triggered_call_logs_summary_and_technique(self):
        log = RecordingLogger()
        events = [context("rules")] + chat("old alpha", "new charlie")
        out = run(events, FakeClient(reply="- s"), compaction_logger=log, keep_recent=1,
                  mechanism="structured", pin_context_events=True)
        (entry,) = log.entries
        self.assertTrue(entry["triggered"])
        self.assertEqual(entry["technique"], "structured+pinned_context")
        self.assertEqual(entry["summary_text"], "- s")
        self.assertEqual(entry["post_text"], out)
        self.assertEqual(entry["pinned_text"], format_blackboard_events_for_prompt(events[:1]))

    def test_logger_failure_does_not_break_compaction(self):
        events = chat("old alpha", "new charlie")
        out = run(events, FakeClient(reply="- s"), compaction_logger=BrokenLogger(), keep_recent=1)
        self.assertIn("[Summary of earlier conversation]\n- s", out)


if __name__ == "__main__":
    unittest.main()
