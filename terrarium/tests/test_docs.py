"""
Checks that the component reference docs still match the code.

These docs state defaults, symbol names and rendered output. All of that can
drift silently when the code changes, so each claim is parsed back out of the
markdown and compared against the real thing rather than restated here.

What this cannot check is prose about rationale and trade-offs, which still
needs a human read.
"""

import importlib
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENV_DOC = "terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_environment.md"
COMPACTION_DOC = "terrarium/compaction/README.md"
PERSONAS_DOC = "terrarium/personas/README.md"
COMPONENT_DOCS = [ENV_DOC, COMPACTION_DOC, PERSONAS_DOC]
ALL_DOCS = COMPONENT_DOCS + ["README.md"]

# "- `key` (default `x`): description" / "- `key`: description"
BULLET = re.compile(r"^- `([^`]+)`(?:\s*\(([^)]*)\))?\s*:\s*(.*)$")


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def section(markdown, heading_contains):
    """Return the text of the `## ...` section whose heading matches."""
    out, keeping = [], False
    for line in markdown.splitlines():
        if line.startswith("## "):
            if keeping:
                break
            keeping = heading_contains.lower() in line.lower()
            continue
        if keeping:
            out.append(line)
    return "\n".join(out)


def bullets(text):
    """Parse '- `key` (paren): description' lines into (key, paren, description)."""
    found = []
    for line in text.splitlines():
        m = BULLET.match(line.strip())
        if m:
            found.append((m.group(1), m.group(2) or "", m.group(3)))
    return found


def documented_default(paren):
    """Pull `x` out of 'default `x`'; return None when the default is prose."""
    m = re.search(r"default\s+`([^`]+)`", paren)
    return m.group(1) if m else None


class DocLinks(unittest.TestCase):
    def test_relative_links_resolve(self):
        broken = []
        for doc in ALL_DOCS:
            base = os.path.dirname(os.path.join(REPO, doc))
            for _text, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", read(doc)):
                if link.startswith(("http", "#", "mailto:")):
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(base, link))):
                    broken.append(f"{doc} -> {link}")
        self.assertEqual(broken, [], f"broken relative links: {broken}")

    def test_referenced_source_files_exist(self):
        missing = []
        for doc in ALL_DOCS:
            for path in re.findall(
                r"`((?:terrarium|examples)/[\w./-]+\.(?:py|md|yaml))`", read(doc)
            ):
                if not os.path.exists(os.path.join(REPO, path)):
                    missing.append(f"{doc} -> {path}")
        self.assertEqual(missing, [], f"docs name files that do not exist: {missing}")

    def test_implementation_blocks_list_real_files(self):
        """Each component doc opens with an Implementation list; those must resolve."""
        for doc in COMPONENT_DOCS:
            head = read(doc).split("## ")[0]
            self.assertIn("Implementation:", head, f"{doc} has no Implementation block")
            paths = re.findall(r"`((?:terrarium|examples)/[\w./-]+)`", head)
            self.assertGreaterEqual(len(paths), 3, f"{doc} Implementation block looks empty")
            for p in paths:
                self.assertTrue(
                    os.path.exists(os.path.join(REPO, p)), f"{doc} -> missing {p}"
                )


class EnvironmentDoc(unittest.TestCase):
    def setUp(self):
        self.doc = read(ENV_DOC)
        self.src = read(
            "terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_env.py"
        )

    def test_documented_defaults_match_code(self):
        checked, mismatches = 0, []
        for key, paren, _desc in bullets(section(self.doc, "Configuration reference")):
            documented = documented_default(paren)
            if documented is None or not re.fullmatch(r"-?\d+(\.\d+)?", documented):
                continue
            m = re.search(rf'env_config\.get\("{re.escape(key)}",\s*([^)]+?)\)', self.src)
            if m is None:
                mismatches.append(f"{key}: documented {documented}, no env_config.get in code")
                continue
            actual = m.group(1).strip()
            checked += 1
            if actual != documented:
                mismatches.append(f"{key}: doc={documented} code={actual}")
        self.assertEqual(mismatches, [], f"env defaults drifted: {mismatches}")
        self.assertGreaterEqual(checked, 7, "expected to check at least 7 numeric defaults")

    def test_pressure_level_thresholds_match(self):
        """Probe each band the doc defines; expectations come from the doc."""
        env = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        cls = env.IsThisSeatTakenEnvironment

        class Fake:
            pass

        bands, lower = [], 0.0
        for key, _paren, desc in bullets(section(self.doc, "Actions, tools, and phases")):
            m = re.match(r"([<>]=?)\s*([\d.]+)$", key.strip())
            if not m:
                continue
            op, value = m.group(1), float(m.group(2))
            label = desc.strip().strip("`")
            probe = (lower + value) / 2 if op == "<" else value + 1.0
            bands.append((probe, label, key))
            lower = value
        self.assertEqual(len(bands), 4, f"expected 4 pressure bands, parsed {len(bands)}")

        f = Fake()
        for probe, label, key in bands:
            f.agent_state = {"a": {"social_pressure": probe * 2.0, "base_tolerance": 2.0}}
            actual = cls._pressure_level(f, "a")
            self.assertEqual(
                actual, label,
                f"ratio {probe} (band {key}): code says {actual!r}, doc says {label!r}",
            )

    def test_seat_role_examples_match(self):
        env = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        cls = env.IsThisSeatTakenEnvironment

        class Fake:
            pass

        found = 0
        for line in self.doc.splitlines():
            m = re.match(r"\s*(\w+),\s*cols=(\d+)\s*->\s*(.+)$", line)
            if not m:
                continue
            scenario, cols = m.group(1), int(m.group(2))
            expected = [r.strip() for r in m.group(3).split(",")]
            f = Fake()
            f.scenario_type = scenario
            actual = [cls._seat_role(f, 0, c, 3, cols) for c in range(cols)]
            self.assertEqual(
                actual, expected,
                f"{scenario} cols={cols}: code gives {actual}, doc says {expected}",
            )
            found += 1
        self.assertGreaterEqual(found, 3, "expected at least 3 seat-role examples")

    def test_documented_config_keys_are_all_read(self):
        pkg = "terrarium/environments/dcops/is_this_seat_taken"
        blob = read("terrarium/environments/abstract_environment.py")
        for root, _dirs, files in os.walk(os.path.join(REPO, pkg)):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    blob += read(os.path.relpath(os.path.join(root, f), REPO))

        unread = []
        for key, _paren, _desc in bullets(section(self.doc, "Configuration reference")):
            if key == "name" or not re.fullmatch(r"[a-z_]+", key):
                continue
            if f'"{key}"' not in blob:
                unread.append(key)
        self.assertEqual(unread, [], f"doc lists config keys nothing reads: {unread}")


class CompactionDoc(unittest.TestCase):
    def setUp(self):
        self.doc = read(COMPACTION_DOC)
        self.seq = read("terrarium/communication_protocols/sequential.py")
        self.compactor = importlib.import_module("terrarium.compaction.compactor")

    def test_documented_mechanisms_match_code(self):
        documented = {k for k, _p, _d in bullets(section(self.doc, "Mechanisms"))}
        documented = {k for k in documented if re.fullmatch(r"[a-z_]+", k)}
        self.assertEqual(
            documented, set(self.compactor.MECHANISMS),
            f"doc mechanisms {sorted(documented)} != code {sorted(self.compactor.MECHANISMS)}",
        )

    def test_documented_defaults_match_code(self):
        mismatches, checked = [], 0
        for key, paren, _desc in bullets(section(self.doc, "Configuration reference")):
            documented = documented_default(paren)
            if documented is None:
                continue
            m = re.search(
                rf'_compaction_config\.get\("{re.escape(key)}",\s*([^)]+?)\)', self.seq
            )
            if m is None:
                continue
            actual = m.group(1).strip().strip('"')
            checked += 1
            norm = {"false": "False", "true": "True"}
            expected = norm.get(documented.lower(), documented).strip('"')
            if actual != expected:
                mismatches.append(f"{key}: doc={documented} code={actual}")
        self.assertEqual(mismatches, [], f"compaction defaults drifted: {mismatches}")
        self.assertGreaterEqual(checked, 5, "expected to check at least 5 compaction defaults")

    def test_evict_kinds_and_markers_match(self):
        self.assertEqual(set(self.compactor.DEFAULT_EVICT_KINDS), {"action_executed"})
        self.assertIn("`action_executed`", self.doc)
        for marker in self.compactor._EXTRACTIVE_MARKERS:
            self.assertIn(
                f"`{marker}`", self.doc,
                f"extractive marker {marker!r} is in the code but not the doc",
            )

    def test_documented_max_tokens_matches(self):
        m = re.search(r"`max_tokens` \((\d+)\)", self.doc)
        self.assertIsNotNone(m, "doc no longer states the summary token cap")
        self.assertEqual(int(m.group(1)), self.compactor._MAX_SUMMARY_TOKENS)

    def test_opt_in_claim_holds(self):
        self.assertIn("self.compaction_enabled = bool(_compaction_config)", self.seq)
        self.assertIn("if not self.compaction_enabled:", self.seq)
        self.assertIn("format_blackboard_events_for_prompt(", self.seq)


class PersonasDoc(unittest.TestCase):
    def setUp(self):
        self.doc = read(PERSONAS_DOC)
        self.personas = importlib.import_module("terrarium.personas")
        self.persona_mod = importlib.import_module("terrarium.personas.persona")

    def test_documented_api_symbols_exist(self):
        missing = []
        api = bullets(section(self.doc, "API reference"))
        self.assertGreaterEqual(len(api), 5, "API reference section looks empty")
        for key, _paren, _desc in api:
            name = key.split("(")[0].split(".")[0]
            if not hasattr(self.personas, name):
                missing.append(name)
        for const in re.findall(r"`([A-Z][A-Z_]{2,})`", self.doc):
            # domain codes are trait keys, not exported symbols -- check them
            # against DOMAINS instead
            if const in self.personas.DOMAINS:
                continue
            if not hasattr(self.personas, const):
                missing.append(const)

        # every domain code the doc names must really be a domain
        named_domains = set(re.findall(r"`(EXT|AGR|CON|NEU|OPE)`", self.doc))
        self.assertEqual(
            named_domains, set(self.personas.DOMAINS),
            f"doc names domains {sorted(named_domains)}, code has {sorted(self.personas.DOMAINS)}",
        )
        self.assertEqual(missing, [], f"doc names symbols that do not exist: {sorted(set(missing))}")

    def test_qualifier_levels_match_code(self):
        documented = {}
        for key, _paren, desc in bullets(section(self.doc, "Trait levels")):
            if key.isdigit():
                documented[int(key)] = desc.strip().strip("`")
        self.assertEqual(len(documented), 9, f"expected 9 levels, parsed {sorted(documented)}")

        for level, rendering in documented.items():
            got = self.persona_mod.qualify(level, "LOW", "HIGH")
            if "contributes nothing" in rendering:
                self.assertIsNone(got, f"level {level} should contribute nothing")
                continue
            expected = rendering.replace("<low>", "LOW").replace("<high>", "HIGH")
            self.assertEqual(got, expected, f"level {level}: code={got!r} doc={expected!r}")

    def test_preset_list_matches_code(self):
        mismatches = []
        parsed = bullets(section(self.doc, "Presets"))
        documented_names = set()
        for key, paren, desc in parsed:
            if not paren.startswith("`") or "," not in desc and "EXT" not in desc:
                continue
            documented_names.add(key)
            if key not in self.personas.PRESETS:
                mismatches.append(f"{key}: documented but not in PRESETS")
                continue
            traits = self.personas.PRESETS[key].traits
            documented_levels = {
                d.strip().split()[0]: int(d.strip().split()[1])
                for d in desc.split(",") if len(d.strip().split()) == 2
            }
            for domain in ("EXT", "AGR", "CON", "NEU", "OPE"):
                expected = documented_levels.get(domain, 5)
                actual = traits.get(domain, 5)
                if actual != expected:
                    mismatches.append(f"{key}.{domain}: doc={expected} code={actual}")
            constant = paren.strip("`")
            if getattr(self.personas, constant, None) is not self.personas.PRESETS[key]:
                mismatches.append(f"{key}: constant {constant} does not point at the preset")
        self.assertEqual(mismatches, [], f"preset list drifted: {mismatches}")
        self.assertEqual(
            documented_names, set(self.personas.PRESETS),
            f"doc presets {sorted(documented_names)} != code {sorted(self.personas.PRESETS)}",
        )

    def test_rendered_examples_match_actual_output(self):
        build = self.personas.build_trait_clause
        Persona = self.personas.Persona
        doc_flat = " ".join(self.doc.split())

        self.assertEqual(build(Persona(traits={"Trust": 9})), "I'm extremely trustful.")
        self.assertIn("I'm extremely trustful.", self.doc)

        for label, persona in (
            ("compact", Persona(traits={"EXT": 9})),
            ("full", Persona(traits={"EXT": 9}, verbosity="full")),
        ):
            text = " ".join(build(persona).split())
            self.assertIn(
                text, doc_flat,
                f"the {label} EXT=9 example in the doc is not what the code renders:\n{text}",
            )

    def test_quick_start_example_is_real_output(self):
        prompt = self.personas.build_persona_prompt(
            self.personas.PRESETS["diplomat"], task="You are seated in row 2."
        )
        self.assertIn(
            " ".join(prompt.split()), " ".join(self.doc.split()),
            f"the quick-start example does not match real output:\n{prompt}",
        )

    def test_persona_precedence_claim(self):
        import inspect
        from terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env import (
            IsThisSeatTakenEnvironment,
        )
        src = inspect.getsource(IsThisSeatTakenEnvironment._resolve_personas)
        self.assertLess(src.index("if persona_map"), src.index("if persona_name"))


class ObservationDoc(unittest.TestCase):
    """Section 6 of the env doc: what each agent is and is not told."""

    ENV_MOD = "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
    PROMPTS_MOD = "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_prompts"

    def setUp(self):
        self.doc = read(ENV_DOC)
        self.obs = section(self.doc, "What agents observe")
        self.assertTrue(self.obs.strip(), "env doc has no 'What agents observe' section")
        self.env_mod = importlib.import_module(self.ENV_MOD)
        self.E = self.env_mod.IsThisSeatTakenEnvironment
        self.P = importlib.import_module(self.PROMPTS_MOD).IsThisSeatTakenPrompts

    def _bare_env(self, **state):
        import random
        env = self.E.__new__(self.E)
        env.rng = random.Random(0)
        env.max_iterations = 10
        env.scenario_type = "airplane"
        env.seats = {}
        base = {
            "satisfaction_score": 0.0, "last_instant_reward": 0.0,
            "base_tolerance": 2.0, "social_pressure": 0.0,
            "pressure_requests": 0, "pressure_complaints": 0,
            "current_seat": None, "settled": False, "pending_reaction": False,
            "preference_profile": {},
        }
        base.update(state)
        env.agent_state = {"a": base}
        return env

    def _bullet_rules(self, name):
        """Parse "`< 1.2` is `low ...`" style rules from a derived-signal bullet."""
        text = " ".join(section(self.doc, "What agents observe").split())
        m = re.search(rf"- `{re.escape(name)}`:(.*?)(?= - `|###|$)", text)
        self.assertIsNotNone(m, f"no derived-signal bullet for {name}")
        body = m.group(1)
        rules = [(op, float(v), label) for op, v, label in
                 re.findall(r"`([<>]=?)\s*([\d.]+)` is `([^`]+)`", body)]
        other = re.search(r"otherwise `([^`]+)`", body)
        return rules, (other.group(1) if other else None)

    def test_tolerance_description_bands(self):
        rules, otherwise = self._bullet_rules("comfort_signal.tolerance_description")
        self.assertEqual(len(rules), 2, f"expected 2 tolerance bands, parsed {rules}")
        # probe just either side of each documented edge, not band midpoints, so a
        # shifted boundary is caught even when the labels are unchanged
        labels = [label for _op, _v, label in rules] + [otherwise]
        probes = []
        for i, (_op, value, label) in enumerate(rules):
            probes.append((value - 0.01, label))
            probes.append((value + 0.01, labels[i + 1]))
        for tol, label in probes:
            env = self._bare_env(base_tolerance=tol)
            got = env.build_agent_context("a", "planning", 1)["comfort_signal"]["tolerance_description"]
            self.assertEqual(got, label, f"tolerance {tol}: code {got!r}, doc {label!r}")

    def test_time_pressure_bands(self):
        rules, otherwise = self._bullet_rules("comfort_signal.time_pressure")
        self.assertEqual(len(rules), 2, f"expected 2 time-pressure bands, parsed {rules}")
        # rules are ">= edge is label", highest edge first; below the lowest edge is
        # `otherwise`. Probe just above and just below every edge.
        ordered = sorted(rules, key=lambda r: r[1])  # ascending edges
        cases = []
        for i, (_op, value, label) in enumerate(ordered):
            below = otherwise if i == 0 else ordered[i - 1][2]
            cases.append((value + 0.01, label))
            cases.append((value - 0.01, below))
        for fraction, label in cases:
            env = self._bare_env()
            # iteration is fraction * max_iterations (10)
            got = env.build_agent_context("a", "planning", fraction * 10)["comfort_signal"]["time_pressure"]
            self.assertEqual(got, label, f"progress {fraction}: code {got!r}, doc {label!r}")

    def test_visible_seat_radius(self):
        m = re.search(r"Manhattan distance (\d+)", self.obs)
        self.assertIsNotNone(m, "doc no longer states the visibility radius")
        radius = int(m.group(1))
        Seat = self.env_mod.Seat
        env = self._bare_env(current_seat="seat_2_2")
        env.seats = {f"seat_{r+1}_{c+1}": Seat(f"seat_{r+1}_{c+1}", r, c, "middle")
                     for r in range(3) for c in range(3)}
        seated = {s["seat_id"] for s in env.build_agent_context("a", "planning", 1)["visible_seats"]}
        expected = {sid for sid, st in env.seats.items() if abs(st.row - 1) + abs(st.col - 1) <= radius}
        self.assertEqual(seated, expected, "visible seats for a seated agent do not match the documented radius")

        env.agent_state["a"]["current_seat"] = None
        standing = env.build_agent_context("a", "planning", 1)["visible_seats"]
        self.assertEqual(len(standing), 9, "doc says a standing agent sees every seat")

    def test_hard_requirements_are_removed(self):
        """Hard requirements were removed: no generation, no scoring, no prompt or doc text."""
        import inspect

        for fn in (self.E._build_agent_state, self.E._compute_instant_reward):
            self.assertNotIn("hard_", inspect.getsource(fn), f"{fn.__name__} still references hard requirements")
        prompts = read("terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_prompts.py")
        self.assertNotIn("hard requirement", prompts.lower(), "system prompt still mentions hard requirements")
        self.assertNotIn("hard_", self.doc, "env doc still describes hard requirements")

        # every preferred role is worth the same +1 again; nothing silently zeroes the second one
        Seat = self.env_mod.Seat

        def reward(role):
            env = self.E.__new__(self.E)
            env.agent_names = ["a"]
            env.seats = {"s1": Seat("s1", 0, 0, role, occupied_by="a")}
            env.agent_state = {"a": {
                "current_seat": "s1", "public_traits": {},
                "preference_profile": {"preferred_roles": ["window", "aisle"],
                                       "hard_required_role": "window"},  # stale key must be ignored
                "social_pressure": 0, "pressure_complaints": 0, "pressure_requests": 0,
                "base_tolerance": 9,
            }}
            return env._compute_instant_reward("a")

        self.assertEqual((reward("window"), reward("aisle"), reward("middle")), (1.0, 1.0, 0.0))

    def test_recall_listed_only_with_retrieval(self):
        self.assertIn("`recall` is\n  listed only when `llm.compaction.retrieval_enabled` is true", self.doc)

        def system_prompt(retrieval):
            pr = self.P.__new__(self.P)
            pr.env = type("Env", (), {"scenario_type": "airplane"})()
            pr.retrieval_enabled = retrieval
            pr.persona_by_agent = {}
            pr.tool_instruction_data = None
            return pr.get_system_prompt("a")

        self.assertIn("recall(", system_prompt(True))
        self.assertNotIn("recall(", system_prompt(False))

    def test_user_prompt_example_is_real_output(self):
        block = self.obs.split("Example (planning phase):", 1)[1].split("```", 2)[1].strip("\n")

        class Fake:
            pass

        ctx = {
            "phase": "planning", "iteration": 2, "max_iterations": 10, "current_seat": "seat_1_2",
            "preference_profile": {"preferred_roles": ["window"], "avoided_roles": ["middle"],
                                   "preferred_neighbors": ["agent_3"]},
            "comfort_signal": {"label": "uncomfortable", "tolerance_description": "moderate",
                               "time_pressure": "low"},
            "social_pressure_level": "mild", "pending_reaction": True,
            "neighbor_observations": [{"agent_id": "agent_1", "seat_id": "seat_1_1",
                                       "public_traits": {"loudness": 0.82, "talkativeness": 0.4,
                                                            "scent": 0.11}}],
            "visible_seats": [{"seat_id": "seat_1_2", "occupied": True},
                              {"seat_id": "seat_2_2", "occupied": False},
                              {"seat_id": "seat_1_3", "occupied": False}],
        }
        rendered = self.P._get_user_prompt_impl(Fake(), "agent_2", ctx, {})
        self.assertIn(block, rendered, "the example user prompt is not what the prompt builder renders")

    def test_action_event_example_is_real_output(self):
        """Rebuild the example from a real move, not from the doc's own payload.

        The event is produced by _move_agent and Megaboard.log_action_to_blackboards,
        then formatted, so both the content (seat ids) and the format are checked.
        """
        import json
        from terrarium.core.blackboard import Megaboard, format_blackboard_events_for_prompt as fmt

        m = re.search(r"^\[\d+\] \[action_executed\] (\S+) payload=(.*)$", self.obs, re.M)
        self.assertIsNotNone(m, "doc no longer shows an action_executed example")
        agent, doc_payload = m.group(1), json.loads(m.group(2))
        doc_params = doc_payload["action_params"]
        doc_result = doc_payload["details"]["result"]

        Seat = self.env_mod.Seat
        env = self.E.__new__(self.E)
        env.move_cost, env.total_moves = -0.5, 0
        env.seats = {
            doc_result["from_seat"]: Seat(doc_result["from_seat"], 0, 0, "middle", occupied_by=agent),
            doc_result["to_seat"]: Seat(doc_result["to_seat"], 0, 1, "aisle"),
        }
        env.agent_state = {agent: {"current_seat": doc_result["from_seat"], "satisfaction_score": 0.0,
                                   "settled": False, "pending_reaction": False}}
        real_result = env._move_agent(agent, doc_params["seat_id"])

        posted = {}

        class Board:
            agents = [agent]

        class FakeMegaboard:
            blackboards = [Board()]

            def post(self, **kwargs):
                posted.update(kwargs)

        Megaboard.log_action_to_blackboards(FakeMegaboard(), agent, dict(doc_params), real_result)
        rendered = fmt([{"kind": posted["kind"], "agent": agent, "payload": posted["payload"]}])
        self.assertEqual(
            rendered.split("] ", 1)[1], m.group(0).split("] ", 1)[1],
            "the action_executed example does not match a real move event",
        )

    def test_initial_seating_is_logged_not_posted(self):
        import inspect
        import json
        import tempfile
        import types

        self.assertIn("initial_seating.json", self.obs)
        flat = " ".join(self.obs.split())
        self.assertIn("The starting layout is **not** posted.", flat,
                      "doc must state that the seat map is not posted to channels")
        withheld = " ".join(section(self.doc, "What agents observe")
                            .split("### Information agents do not receive", 1)[1].split())
        self.assertIn("the starting seating map, beyond their own seat and their neighbours", withheld,
                      "doc must list the seat map among information agents do not receive")
        src = inspect.getsource(self.E.async_init)
        self.assertNotIn("post_system_message", src, "doc says the seat map is not posted to channels")
        self.assertNotIn("Initial seating", src)

        Seat = self.env_mod.Seat
        env = self.E.__new__(self.E)
        env.seats = {
            "s1": Seat("s1", 0, 0, "window", occupied_by="agent_0"),
            "s2": Seat("s2", 0, 1, "aisle"),
            "s3": Seat("s3", 0, 2, "middle", occupied_by="agent_1"),
        }
        import asyncio
        from unittest import mock

        from terrarium.environments.abstract_environment import AbstractEnvironment

        posts = []

        class RecordingProtocol:
            async def get_all_blackboard_ids(self):
                return [0, 1]

            async def post_system_message(self, *args, **kwargs):
                posts.append((args, kwargs))

        with tempfile.TemporaryDirectory() as d:
            env.tool_logger = types.SimpleNamespace(log_dir=d)
            env.communication_protocol = RecordingProtocol()
            # run the real async_init; only the framework-level network setup is stubbed
            with mock.patch.object(AbstractEnvironment, "async_init", new=mock.AsyncMock()):
                asyncio.run(env.async_init())
            path = os.path.join(d, "initial_seating.json")
            self.assertTrue(os.path.exists(path), "async_init did not write initial_seating.json")
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"agent_0": "s1", "agent_1": "s3"})
        self.assertEqual(posts, [], f"async_init posted to channels: {posts}")

    def test_own_tool_result_is_not_shown_to_the_model(self):
        """Doc: a successful env tool call ends the turn, so the result is never read."""
        import inspect
        from terrarium.agents import base
        execute = inspect.getsource(base.BaseAgent._execute_tool_call)
        loop = inspect.getsource(base.BaseAgent._multi_step_response_generation)
        self.assertIn("self._env_state_committed = True", execute)
        self.assertIn("if self._env_state_committed:", loop)
        # the follow-up model call (the one that would read tool results) is the one
        # inside the step loop, not the initial call before it
        loop_start = loop.index("for step in range")
        self.assertLess(loop.index("if self._env_state_committed:", loop_start),
                        loop.index("self.client.generate_response", loop_start),
                        "the turn no longer ends before the model sees the tool result")

    def test_social_asks_skip_private_channels(self):
        import inspect
        tools = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_tools"
        ).IsThisSeatTakenTools
        self.assertIn("to non-private channels only", self.obs)
        self.assertIn("is_private_channel", inspect.getsource(tools._post_social_message))
        self.assertNotIn("log_action_to_blackboards",
                         inspect.getsource(tools.execute_action).split("EXECUTION_PHYSICAL_ACTIONS")[0],
                         "social asks are now broadcast as action events")


class ActionCostDoc(unittest.TestCase):
    """Section 5 execution actions: costs and side effects, checked by running them."""

    def setUp(self):
        import random
        self.doc = read(ENV_DOC)
        self.actions = section(self.doc, "Actions, tools, and phases")
        self.flat = " ".join(self.actions.split())
        mod = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        self.E, self.Seat, self.random = mod.IsThisSeatTakenEnvironment, mod.Seat, random

    def _env(self):
        env = self.E.__new__(self.E)
        env.rng = self.random.Random(0)
        env.move_cost, env.social_action_cost = -0.5, -0.3
        env.total_moves, env.time_step, env.max_iterations = 0, 0, 10
        env.scenario_type = "airplane"
        env.agent_names = ["a", "b"]
        env.seats = {
            "s1": self.Seat("s1", 0, 0, "window", neighbors=["s2"], occupied_by="a"),
            "s2": self.Seat("s2", 0, 1, "middle", neighbors=["s1", "s3"], occupied_by="b"),
            "s3": self.Seat("s3", 0, 2, "aisle", neighbors=["s2"]),
        }

        def state(seat):
            return {"current_seat": seat, "satisfaction_score": 0.0, "last_instant_reward": 0.0,
                    "settled": True, "pending_reaction": False, "base_tolerance": 9.0,
                    "social_pressure": 0.0, "pressure_requests": 0, "pressure_complaints": 0,
                    "preference_profile": {},
                    "public_traits": {"loudness": 0, "scent": 0, "talkativeness": 0}}

        env.agent_state = {"a": state("s1"), "b": state("s2")}
        return env

    def _run(self, action, args):
        env = self._env()
        result = env._apply_agent_action("a", action, args)
        return env, result, env.agent_state["a"]["satisfaction_score"]

    def _documented_cost(self, bullet_key):
        descs = {k: d for k, _p, d in bullets(self.actions)}
        self.assertIn(bullet_key, descs, f"section 5 has no bullet for {bullet_key}")
        # a bullet's description continues on indented lines; take the whole paragraph
        start = self.flat.index(f"- `{bullet_key}`:")
        nxt = self.flat.find(" - `", start + 1)
        text = self.flat[start: nxt if nxt != -1 else None]
        if "No cost" in text:
            return 0.0, text
        if "costs `move_cost`" in text:
            return -0.5, text
        self.fail(f"cannot tell the documented cost of {bullet_key}: {text!r}")

    def test_successful_move_cost(self):
        expected, text = self._documented_cost("move(seat_id)")
        env, result, cost = self._run("move", {"seat_id": "s3"})
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(cost, expected, msg=f"move cost: code {cost}, doc {expected}")
        self.assertIn("increments `total_moves`", text)
        self.assertEqual(env.total_moves, 1)

    def test_failed_move_into_occupied_seat(self):
        charged = re.search(r"occupied seat fails but still costs `move_cost`", self.flat)
        free = re.search(r"occupied seat fails at no cost", self.flat)
        self.assertTrue(charged or free, "doc no longer says what a failed move costs")
        env, result, cost = self._run("move", {"seat_id": "s2"})
        self.assertEqual(result["status"], "failed")
        expected = -0.5 if charged else 0.0
        self.assertAlmostEqual(cost, expected, msg=f"failed move: code {cost}, doc {expected}")
        if "without incrementing `total_moves`" in self.flat:
            self.assertEqual(env.total_moves, 0)

    def test_stand_cost(self):
        expected, text = self._documented_cost("stand()")
        env, result, cost = self._run("stand", {})
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(cost, expected, msg=f"stand cost: code {cost}, doc {expected}")
        if "does not increment `total_moves`" in text:
            self.assertEqual(env.total_moves, 0)

    def test_settle_cost(self):
        expected, _text = self._documented_cost("settle()")
        _env, result, cost = self._run("settle", {})
        self.assertEqual(result["status"], "success")
        self.assertAlmostEqual(cost, expected, msg=f"settle cost: code {cost}, doc {expected}")

    def test_rejected_action_clears_settled(self):
        self.assertRegex(
            self.flat,
            r"clears the agent's `settled` flag before it is validated",
            "doc no longer states that a rejected action un-settles the agent",
        )
        env, result, cost = self._run("move", {})
        self.assertEqual(result["status"], "retry")
        self.assertEqual(cost, 0.0, "a move with no seat_id should cost nothing")
        self.assertFalse(env.agent_state["a"]["settled"], "rejected action did not clear settled")


class TerminationWordingDoc(unittest.TestCase):
    """Agents must be told the real end condition, not 'ends when everyone settles'."""

    NEW = "ends once everyone has settled and nothing is still changing, or when time runs out"

    def test_agents_are_told_the_actual_rule(self):
        import inspect

        tools = read("terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_tools.py")
        prompts = read("terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_prompts.py")
        for old in (
            "The simulation ends when every agent has settled",
            "The simulation ends when every agent has called settle()",
            "the run ends only when ALL agents have settled",
        ):
            self.assertNotIn(old, tools + prompts, f"old termination claim still shown: {old!r}")
        self.assertIn(self.NEW, tools, "settle tool description lacks the corrected wording")
        self.assertEqual(prompts.count(self.NEW), 2, "system and user prompt should both carry it")

        env = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        ).IsThisSeatTakenEnvironment
        done = inspect.getsource(env.done)
        self.assertIn("iteration > self.max_iterations", done, "wording says runs end when time runs out")
        self.assertIn("reward_converged or reward_near_max", done,
                      "wording says settling alone is not enough")

        doc = " ".join(read(ENV_DOC).split())
        self.assertNotIn("both tell agents that the simulation ends", doc,
                         "limitation describes wording that no longer exists")


class GuiInitialSeating(unittest.TestCase):
    GUI = "terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py"

    def setUp(self):
        import importlib.util
        import sys

        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("flask not installed")
        argv = sys.argv
        sys.argv = ["gui", "--log", os.devnull, "--no-browser"]
        try:
            spec = importlib.util.spec_from_file_location("seat_gui_under_test", os.path.join(REPO, self.GUI))
            self.gui = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.gui)
        finally:
            sys.argv = argv

    def test_positions_come_from_initial_seating_json(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "blackboard_0.txt")
            open(log, "w").close()
            self.assertIsNone(self.gui._load_initial_seating(log))
            with open(os.path.join(d, "initial_seating.json"), "w", encoding="utf-8") as f:
                json.dump({"agent_0": "seat_1_1", "agent_1": "seat_1_2"}, f)
            initial = self.gui._load_initial_seating(log)
        pos, _settled = self.gui.infer_positions([], initial)
        self.assertEqual(pos, {"agent_0": "seat_1_1", "agent_1": "seat_1_2"})

        # when both exist, the json is authoritative and a stale chat message is ignored
        import types

        stale = types.SimpleNamespace(etype="context", content="Initial seating: agent_0→seat_9_9, agent_1→seat_9_8")
        pos, _settled = self.gui.infer_positions([stale], initial)
        self.assertEqual(pos, {"agent_0": "seat_1_1", "agent_1": "seat_1_2"},
                         "a chat seat map overrode initial_seating.json")

    def test_older_logs_still_read_the_chat_message(self):
        import types

        event = types.SimpleNamespace(etype="context", content="Initial seating: agent_0→seat_1_1, agent_1→seat_1_2")
        pos, _settled = self.gui.infer_positions([event], None)
        self.assertEqual(pos, {"agent_0": "seat_1_1", "agent_1": "seat_1_2"})


class TraitCutoffDoc(unittest.TestCase):
    """Section 7: a neighbour's trait costs -1 above effective_tolerance / MAX_BASE_TOLERANCE."""

    def setUp(self):
        self.doc = read(ENV_DOC)
        self.mod = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        self.E, self.Seat = self.mod.IsThisSeatTakenEnvironment, self.mod.Seat
        m = re.search(r"trait_cutoff = effective / ([\d.]+)", self.doc)
        self.assertIsNotNone(m, "doc no longer states the trait cutoff formula")
        self.divisor = float(m.group(1))

    def _reward(self, tolerance, trait, pressure=0.0, complaints=0):
        env = self.E.__new__(self.E)
        env.agent_names = ["a", "b"]
        env.seats = {
            "s1": self.Seat("s1", 0, 0, "middle", neighbors=["s2"], occupied_by="a"),
            "s2": self.Seat("s2", 0, 1, "middle", neighbors=["s1"], occupied_by="b"),
        }

        def state(seat, tol, traits):
            return {"current_seat": seat, "preference_profile": {}, "public_traits": traits,
                    "base_tolerance": tol, "social_pressure": pressure if seat == "s1" else 0.0,
                    "pressure_complaints": complaints if seat == "s1" else 0,
                    "pressure_requests": 0}

        quiet = {"loudness": 0.0, "scent": 0.0, "talkativeness": 0.0}
        env.agent_state = {"a": state("s1", tolerance, quiet),
                           "b": state("s2", 9.0, dict(quiet, loudness=trait))}
        return env._compute_instant_reward("a")

    def test_divisor_matches_code_and_tolerance_range(self):
        import inspect
        self.assertEqual(self.divisor, self.mod.MAX_BASE_TOLERANCE)
        self.assertIn("uniform(1.0, MAX_BASE_TOLERANCE)", inspect.getsource(self.E._build_agent_state),
                      "base_tolerance's upper bound must be the same constant as the divisor")
        self.assertIn(f"uniform(1.0, {self.divisor:g})", self.doc)

    def test_penalty_fires_just_above_the_cutoff_only(self):
        flat = " ".join(self.doc.split())
        self.assertIn("A neighbour's trait costs the agent `-1` when it is above `trait_cutoff`.", flat,
                      "doc no longer states the trait penalty rule")
        cutoff = 1.75 / self.divisor
        self.assertEqual(self._reward(1.75, cutoff + 0.01), -1.0, "trait above cutoff did not cost")
        self.assertEqual(self._reward(1.75, cutoff - 0.01), 0.0, "trait below cutoff cost anyway")

    def test_pressure_lowers_the_cutoff(self):
        tol, pressure, complaints = 1.75, 0.9, 1
        degraded = tol - pressure * 0.15 - complaints * 0.05
        trait = (degraded / self.divisor + tol / self.divisor) / 2  # between the two cutoffs
        self.assertEqual(self._reward(tol, trait), 0.0, "should not cost without pressure")
        self.assertEqual(self._reward(tol, trait, pressure, complaints), -1.0,
                         "pressure did not lower the cutoff")

    def test_tolerance_words_point_the_right_way(self):
        """The tolerance words shown to agents must match the neighbour mechanics:
        the 'easily bothered' band has the lower trait cutoff."""
        env = self.E.__new__(self.E)
        env.max_iterations, env.scenario_type, env.seats = 10, "airplane", {}

        def described(tol):
            env.agent_state = {"a": {"satisfaction_score": 0.0, "last_instant_reward": 0.0,
                                     "base_tolerance": tol, "social_pressure": 0.0,
                                     "pressure_requests": 0, "pressure_complaints": 0,
                                     "current_seat": None, "settled": False,
                                     "pending_reaction": False, "preference_profile": {}}}
            return env.build_agent_context("a", "planning", 1)["comfort_signal"]["tolerance_description"]

        low, high = described(1.05), described(2.45)
        self.assertIn("bother you easily", low)
        self.assertIn("rarely bother you", high)
        self.assertLess(1.05 / self.divisor, 2.45 / self.divisor)
        # the easily-bothered agent really does pay more trait penalties for the same neighbour
        trait = (1.05 / self.divisor + 2.45 / self.divisor) / 2
        self.assertEqual(self._reward(1.05, trait), -1.0)
        self.assertEqual(self._reward(2.45, trait), 0.0)
        for stale in ("picky", "settle easily"):
            self.assertNotIn(stale, low + high)

    def test_cutoff_range_stated_in_doc(self):
        flat = " ".join(self.doc.split())
        m = re.search(r"cutoff runs from ([\d.]+) for the least tolerant agent to ([\d.]+)", flat)
        self.assertIsNotNone(m, "doc no longer states the cutoff range")
        self.assertAlmostEqual(float(m.group(1)), 1.0 / self.divisor)
        self.assertAlmostEqual(float(m.group(2)), self.mod.MAX_BASE_TOLERANCE / self.divisor)


class WordsOnlyDoc(unittest.TestCase):
    """Agents see their comfort and neighbours' traits as words: no numbers, no noise."""

    def setUp(self):
        self.doc = read(ENV_DOC)
        self.obs = section(self.doc, "What agents observe")
        env_mod = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        self.E, self.Seat = env_mod.IsThisSeatTakenEnvironment, env_mod.Seat
        self.prompts = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_prompts"
        )

    def test_no_noise_in_what_agents_see(self):
        import inspect
        for fn in (self.E.build_agent_context, self.E._occupied_neighbors):
            self.assertNotIn("self.rng", inspect.getsource(fn), f"{fn.__name__} still adds random noise")
        self.assertNotIn("felt_satisfaction", inspect.getsource(self.E.build_agent_context))
        for stale in ("noisy", "gauss(", "uniform(-0.3", "rough score", "perceived"):
            self.assertNotIn(stale, self.obs, f"section 6 still mentions {stale!r}")
        flat = " ".join(self.obs.split())
        self.assertIn("No noise is added, and exact values are never shown.", flat,
                      "doc must state that trait words carry no noise")
        self.assertEqual(flat.lower().count("noise"), 1, "section 6 mentions noise somewhere else")

    def test_trait_words_match_the_doc(self):
        flat = " ".join(self.obs.split())
        m = re.search(r"Loudness reads, from lowest to highest, (.*?); talkativeness", flat)
        self.assertIsNotNone(m, "doc no longer lists the loudness words")
        documented = tuple(re.findall(r"`([^`]+)`", m.group(1)))
        self.assertEqual(documented, self.prompts._TRAIT_WORDS["loudness"])
        n = len(documented)
        self.assertIn("using five equal bands", flat)
        self.assertEqual(n, 5, "doc says five bands")
        for level, word in enumerate(documented):
            lo, hi = level / n, (level + 1) / n
            self.assertEqual(self.prompts.describe_trait("loudness", lo + 0.001), word)
            self.assertEqual(self.prompts.describe_trait("loudness", hi - 0.001), word)
        self.assertEqual(self.prompts.describe_trait("loudness", 1.0), documented[-1])
        for trait in ("talkativeness", "scent"):
            self.assertEqual(len(self.prompts._TRAIT_WORDS[trait]), n, f"{trait} should use the same steps")

    def test_rendered_prompt_shows_no_numbers_for_comfort_or_traits(self):
        Seat = self.Seat
        env = self.E.__new__(self.E)  # no rng: any leftover noise draw fails here
        env.max_iterations, env.scenario_type = 10, "airplane"
        env.seats = {
            "seat_1_1": Seat("seat_1_1", 0, 0, "window", neighbors=["seat_1_2"], occupied_by="a"),
            "seat_1_2": Seat("seat_1_2", 0, 1, "middle", neighbors=["seat_1_1"], occupied_by="b"),
        }

        def state(seat, traits):
            return {"satisfaction_score": -1.37, "last_instant_reward": -1.0, "base_tolerance": 1.6,
                    "social_pressure": 0.0, "pressure_requests": 0, "pressure_complaints": 0,
                    "current_seat": seat, "settled": False, "pending_reaction": False,
                    "preference_profile": {"preferred_roles": ["aisle"]}, "public_traits": traits}

        env.agent_state = {
            "a": state("seat_1_1", {"loudness": 0.1, "scent": 0.1, "talkativeness": 0.1}),
            "b": state("seat_1_2", {"loudness": 0.93, "scent": 0.07, "talkativeness": 0.55}),
        }
        ctx = env.build_agent_context("a", "planning", 1)

        class Fake:
            pass

        prompt = self.prompts.IsThisSeatTakenPrompts._get_user_prompt_impl(Fake(), "a", ctx, {})
        feel = next(l for l in prompt.splitlines() if l.startswith("**How you feel:**"))
        self.assertIsNone(re.search(r"\d", feel), f"comfort line shows a number: {feel}")
        neighbour = next(l for l in prompt.splitlines() if "b at seat_1_2" in l)
        self.assertIsNone(re.search(r"\d", neighbour.split("—", 1)[1]), f"trait line shows numbers: {neighbour}")
        for words in ("very loud", "somewhat talkative", "barely noticeable scent"):
            self.assertIn(words, neighbour)


class IsolationPrecedenceDoc(unittest.TestCase):
    """An agent that prefers isolation gets no credit for, and is not shown, a preferred neighbour."""

    def setUp(self):
        self.doc = read(ENV_DOC)
        mod = importlib.import_module("terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env")
        self.E, self.Seat = mod.IsThisSeatTakenEnvironment, mod.Seat
        self.P = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_prompts"
        ).IsThisSeatTakenPrompts

    def _reward(self, profile, neighbour_present):
        env = self.E.__new__(self.E)
        env.agent_names = ["a", "b"]
        env.seats = {
            "s1": self.Seat("s1", 0, 0, "middle", neighbors=["s2"], occupied_by="a"),
            "s2": self.Seat("s2", 0, 1, "middle", neighbors=["s1"], occupied_by="b" if neighbour_present else None),
        }
        quiet = {"loudness": 0.0, "scent": 0.0, "talkativeness": 0.0}
        base = {"social_pressure": 0, "pressure_complaints": 0, "pressure_requests": 0, "base_tolerance": 2.5}
        env.agent_state = {
            "a": dict(base, current_seat="s1", public_traits=quiet, preference_profile=profile),
            "b": dict(base, current_seat="s2" if neighbour_present else None, public_traits=quiet,
                      preference_profile={}),
        }
        return env._compute_instant_reward("a")

    def test_doc_states_the_rule(self):
        flat = " ".join(self.doc.split())
        self.assertIn("isolation takes precedence: the preferred neighbour is neither scored nor shown "
                      "to the agent.", flat)
        self.assertIn("a `preferred_neighbor` is adjacent, unless the agent prefers isolation", flat)

    def test_isolation_wins_in_the_reward(self):
        both = {"prefer_isolation": True, "preferred_neighbors": ["b"]}
        self.assertEqual(self._reward(both, neighbour_present=True), -1.0,
                         "preferred neighbour should not offset the isolation penalty")
        self.assertEqual(self._reward(both, neighbour_present=False), 1.0)
        self.assertEqual(self._reward({"preferred_neighbors": ["b"]}, neighbour_present=True), 1.0,
                         "without isolation the preferred neighbour still counts")

    def test_preferred_neighbour_hidden_from_prompt_when_isolated(self):
        class Fake:
            pass

        def rendered(profile):
            ctx = {"phase": "planning", "current_seat": "s1", "preference_profile": profile,
                   "comfort_signal": {}, "social_pressure_level": "none"}
            return self.P._get_user_prompt_impl(Fake(), "a", ctx, {})

        isolated = rendered({"prefer_isolation": True, "preferred_neighbors": ["agent_9"]})
        self.assertNotIn("agent_9", isolated)
        self.assertIn("you prefer having empty seats around you", isolated)
        self.assertIn("you'd rather sit near agent_9", rendered({"preferred_neighbors": ["agent_9"]}))


class EarlyStopDoc(unittest.TestCase):
    """Section 8: joint reward is recorded per iteration, so settled, stable runs stop early."""

    def setUp(self):
        self.doc = read(ENV_DOC)
        self.E = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        ).IsThisSeatTakenEnvironment

    def _env(self):
        env = self.E.__new__(self.E)
        env.agent_names = ["a", "b"]
        env.agent_state = {x: {"settled": True, "satisfaction_score": 1.0, "social_pressure": 0.0,
                               "pressure_complaints": 0, "pressure_requests": 0} for x in env.agent_names}
        env.total_moves, env.group_move_penalty, env.time_step = 0, 0.1, 0
        env.max_iterations, env.scenario_type = 10, "airplane"
        env.convergence_window, env.reward_convergence_threshold = 3, 0.05
        env.joint_reward_history, env.max_joint_reward = [], 100.0
        return env

    def _run(self, env, change_each_iteration=0.0):
        """Mirror base_main: done() at the start of each iteration, log_iteration() at the end."""
        for iteration in range(1, env.max_iterations + 1):
            if env.done(iteration):
                return iteration
            env.agent_state["a"]["satisfaction_score"] += change_each_iteration
            env.log_iteration(iteration)
        return None

    def test_joint_reward_is_pure(self):
        env = self._env()
        env.joint_reward({})
        env.joint_reward({})
        self.assertEqual(env.joint_reward_history, [], "joint_reward() must not write to the history")

    def test_settled_stable_run_stops_at_window_plus_one(self):
        env = self._env()
        self.assertEqual(self._run(env), env.convergence_window + 1)

    def test_changing_reward_does_not_stop_early(self):
        env = self._env()
        self.assertIsNone(self._run(env, change_each_iteration=0.5), "run stopped while reward was changing")

    def test_unsettled_agents_do_not_stop_early(self):
        env = self._env()
        env.agent_state["b"]["settled"] = False
        self.assertIsNone(self._run(env), "run stopped while an agent was unsettled")

    def test_doc_states_recording_and_earliest_stop(self):
        flat = " ".join(section(self.doc, "Termination").split())
        self.assertIn("`log_iteration()` records the joint reward at the end of each iteration", flat)
        self.assertIn("no sooner than the start of iteration `convergence_window + 1`", flat)
        self.assertNotIn("by design", self.doc, "doc makes an unsupported design-intent claim")


class DocumentedCLI(unittest.TestCase):
    def test_documented_flags_exist(self):
        flags = set()
        for doc in COMPONENT_DOCS:
            flags |= set(re.findall(r"(--[a-z][a-z-]+)", read(doc)))
        self.assertGreaterEqual(len(flags), 4, "expected the docs to show CLI flags")

        known = set()
        for entry in (
            "examples/base_main.py",
            "terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py",
        ):
            known |= set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', read(entry)))
        missing = sorted(f for f in flags if f not in known)
        self.assertEqual(missing, [], f"docs show flags nothing defines: {missing}")


class ShippedConfigs(unittest.TestCase):
    def test_no_dead_keys_in_seating_config(self):
        import yaml

        blob = read("examples/base_main.py")
        for root, _dirs, files in os.walk(os.path.join(REPO, "terrarium")):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    blob += read(os.path.relpath(os.path.join(root, f), REPO))

        cfg = yaml.safe_load(read("examples/configs/is_this_seat_taken.yaml"))
        dead = [k for k in cfg["environment"] if f'"{k}"' not in blob]
        self.assertEqual(dead, [], f"shipped config has keys nothing reads: {dead}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
