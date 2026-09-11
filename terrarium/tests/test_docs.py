"""
Checks that the component reference docs still match the code.

These docs state defaults, symbol names and rendered output. All of that can
drift silently when the code changes, so each claim is parsed back out of the
markdown and compared against the real thing rather than restated here.

What this cannot check: prose about rationale and trade-offs. Those need a
human read.
"""

import importlib
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENV_DOC = "terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_environment.md"
COMPACTION_DOC = "terrarium/compaction/README.md"
PERSONAS_DOC = "terrarium/personas/README.md"
ALL_DOCS = [ENV_DOC, COMPACTION_DOC, PERSONAS_DOC, "README.md"]


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def table_rows(markdown, header_contains):
    """Yield cell-lists for the markdown table whose header mentions the given text."""
    rows, in_table = [], False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not in_table:
            if header_contains.lower() in " | ".join(cells).lower():
                in_table = True
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def unbacktick(s):
    return s.strip().strip("`").strip()


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
            for path in re.findall(r"`((?:terrarium|examples)/[\w./-]+\.(?:py|md|yaml))`", read(doc)):
                if not os.path.exists(os.path.join(REPO, path)):
                    missing.append(f"{doc} -> {path}")
        self.assertEqual(missing, [], f"docs name files that do not exist: {missing}")


class EnvironmentDoc(unittest.TestCase):
    """The env doc's config table must match the environment's real defaults."""

    def setUp(self):
        self.doc = read(ENV_DOC)
        self.src = read("terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_env.py")

    def test_documented_defaults_match_code(self):
        # Only rows whose default is a bare number; the rest are prose defaults.
        checked, mismatches = 0, []
        for cells in table_rows(self.doc, "Key | Default"):
            if len(cells) < 2:
                continue
            key, documented = unbacktick(cells[0]), unbacktick(cells[1])
            if not re.fullmatch(r"-?\d+(\.\d+)?", documented):
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
        """Probe each band the doc defines and confirm the code agrees.

        Expectations are derived from the table, not hardcoded, so editing a
        label in the doc without changing _pressure_level fails here.
        """
        env = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        cls = env.IsThisSeatTakenEnvironment

        class Fake:
            pass

        bands, lower = [], 0.0
        for cells in table_rows(self.doc, "Level shown"):
            if len(cells) < 2:
                continue
            bound_text, label = unbacktick(cells[0]), unbacktick(cells[1])
            m = re.match(r"([<≥])\s*([\d.]+)", bound_text)
            self.assertIsNotNone(m, f"unparsed pressure band {bound_text!r}")
            op, value = m.group(1), float(m.group(2))
            # a ratio inside this band
            probe = (lower + value) / 2 if op == "<" else value + 1.0
            bands.append((probe, label, bound_text))
            lower = value
        self.assertEqual(len(bands), 4, f"expected 4 pressure bands, parsed {len(bands)}")

        f = Fake()
        for probe, label, bound_text in bands:
            f.agent_state = {"a": {"social_pressure": probe * 2.0, "base_tolerance": 2.0}}
            actual = cls._pressure_level(f, "a")
            self.assertEqual(
                actual, label,
                f"ratio {probe} (band {bound_text}): code says {actual!r}, doc says {label!r}",
            )

    def test_seat_role_worked_examples_match(self):
        env = importlib.import_module(
            "terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env"
        )
        cls = env.IsThisSeatTakenEnvironment

        class Fake:
            pass

        # parse the fenced "scenario, cols=N -> a, b, c" examples out of the doc
        found = 0
        for line in self.doc.splitlines():
            m = re.match(r"\s*(\w+),\s*cols=(\d+)\s*→\s*(.+)$", line)
            if not m:
                continue
            scenario, cols, expected = m.group(1), int(m.group(2)), m.group(3)
            expected_roles = [r.strip() for r in expected.split(",")]
            f = Fake()
            f.scenario_type = scenario
            actual = [cls._seat_role(f, 0, c, 3, cols) for c in range(cols)]
            self.assertEqual(
                actual, expected_roles,
                f"{scenario} cols={cols}: code gives {actual}, doc says {expected_roles}",
            )
            found += 1
        self.assertGreaterEqual(found, 3, "expected at least 3 worked seat-role examples")

    def test_documented_config_keys_are_all_read(self):
        """The doc's config table should not list a key no code reads."""
        pkg = "terrarium/environments/dcops/is_this_seat_taken"
        sources = []
        for root, _dirs, files in os.walk(os.path.join(REPO, pkg)):
            if "__pycache__" in root:
                continue
            sources += [read(os.path.relpath(os.path.join(root, f), REPO))
                        for f in files if f.endswith(".py")]
        blob = "\n".join(sources) + read("terrarium/environments/abstract_environment.py")

        unread = []
        for cells in table_rows(self.doc, "Key | Default"):
            key = unbacktick(cells[0])
            if key in ("name",) or not re.fullmatch(r"[a-z_]+", key):
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
        documented = {unbacktick(c[0]) for c in table_rows(self.doc, "Mechanism | What it does")}
        self.assertEqual(
            documented, set(self.compactor.MECHANISMS),
            f"doc mechanisms {sorted(documented)} != code {sorted(self.compactor.MECHANISMS)}",
        )

    def test_documented_defaults_match_code(self):
        mismatches, checked = [], 0
        for cells in table_rows(self.doc, "Key | Default | Meaning"):
            if len(cells) < 2:
                continue
            key, documented = unbacktick(cells[0]), unbacktick(cells[1])
            m = re.search(rf'_compaction_config\.get\("{re.escape(key)}",\s*([^)]+?)\)', self.seq)
            if m is None:
                continue  # prose default (e.g. "falls back to llm.provider")
            actual = m.group(1).strip()
            checked += 1
            norm = {"false": "False", "true": "True"}
            doc_norm = norm.get(documented.lower(), documented)
            if actual.strip('"') != doc_norm.strip('"'):
                mismatches.append(f"{key}: doc={documented} code={actual}")
        self.assertEqual(mismatches, [], f"compaction defaults drifted: {mismatches}")
        self.assertGreaterEqual(checked, 5, "expected to check at least 5 compaction defaults")

    def test_evict_kinds_and_markers_match(self):
        self.assertIn("`action_executed`", self.doc)
        self.assertEqual(set(self.compactor.DEFAULT_EVICT_KINDS), {"action_executed"})
        for marker in self.compactor._EXTRACTIVE_MARKERS:
            self.assertIn(
                f"`{marker}`".replace("` `", "`"), self.doc.replace("will `", "will `"),
                f"extractive marker {marker!r} is in the code but not the doc",
            )

    def test_documented_max_tokens_matches(self):
        m = re.search(r"`max_tokens` \((\d+)\)", self.doc)
        self.assertIsNotNone(m, "doc no longer states the summary token cap")
        self.assertEqual(int(m.group(1)), self.compactor._MAX_SUMMARY_TOKENS)

    def test_opt_in_claim_holds(self):
        """Doc says: no llm.compaction block -> compaction disabled."""
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
        for cells in table_rows(self.doc, "Function | Purpose"):
            name = unbacktick(cells[0]).split("(")[0].split(".")[0]
            if not hasattr(self.personas, name):
                missing.append(name)
        for const in re.findall(r"`([A-Z][A-Z_]+)`", self.doc):
            if const in ("EXT", "AGR", "CON", "NEU", "OPE", "DECISIONS", "COMMITMENTS"):
                continue
            if not hasattr(self.personas, const):
                missing.append(const)
        self.assertEqual(missing, [], f"doc names symbols that do not exist: {sorted(set(missing))}")

    def test_qualifier_table_matches_code(self):
        documented = {}
        for cells in table_rows(self.doc, "Level | Rendering"):
            for i in (0, 3):
                if i + 1 >= len(cells):
                    continue
                lvl, rendering = cells[i].strip(), cells[i + 1].strip()
                if lvl.isdigit() and rendering:
                    documented[int(lvl)] = rendering.replace("`", "").strip()
        self.assertTrue(documented, "qualifier table not found")

        for level, rendering in documented.items():
            got = self.persona_mod.qualify(level, "LOW", "HIGH")
            if level == 5:
                self.assertIsNone(got, "level 5 should contribute nothing")
                continue
            expected = rendering.replace("<low>", "LOW").replace("<high>", "HIGH")
            expected = expected.replace(" (bare)", "")
            self.assertEqual(got, expected, f"level {level}: code={got!r} doc={expected!r}")

    def test_preset_table_matches_code(self):
        mismatches = []
        for cells in table_rows(self.doc, "EXT | AGR | CON"):
            name = unbacktick(cells[0])
            if name not in self.personas.PRESETS:
                mismatches.append(f"{name}: documented but not in PRESETS")
                continue
            traits = self.personas.PRESETS[name].traits
            for idx, domain in enumerate(("EXT", "AGR", "CON", "NEU", "OPE"), start=1):
                documented = unbacktick(cells[idx])
                actual = traits.get(domain, 5)
                expected = 5 if documented in ("–", "-", "—", "") else int(documented)
                if actual != expected:
                    mismatches.append(f"{name}.{domain}: doc={documented} code={actual}")
        self.assertEqual(mismatches, [], f"preset table drifted: {mismatches}")
        self.assertEqual(
            len(self.personas.PRESETS), 5,
            "doc says five presets ship; PRESETS has a different count",
        )

    def test_rendered_examples_match_actual_output(self):
        """Every prose example in the doc must be real output, not paraphrase."""
        build = self.personas.build_trait_clause
        Persona = self.personas.Persona

        cases = [
            (Persona(traits={"Trust": 9}), "I'm extremely trustful."),
            (Persona(traits={"A1": 9}), "I'm extremely trustful."),
        ]
        for persona, expected in cases:
            self.assertEqual(build(persona), expected)
            self.assertIn(expected, self.doc, f"doc no longer shows {expected!r}")

        # the compact/full EXT=9 block in section 4
        compact = build(Persona(traits={"EXT": 9}))
        full = build(Persona(traits={"EXT": 9}, verbosity="full"))
        doc_flat = " ".join(self.doc.split())
        for label, text in (("compact", compact), ("full", full)):
            self.assertIn(
                " ".join(text.split()), doc_flat,
                f"the {label} EXT=9 example in the doc is not what the code renders:\n{text}",
            )

    def test_quick_start_example_is_real_output(self):
        prompt = self.personas.build_persona_prompt(
            self.personas.PRESETS["diplomat"], task="You are seated in row 2."
        )
        doc_flat = " ".join(self.doc.split())
        self.assertIn(
            " ".join(prompt.split()), doc_flat,
            f"the quick-start example does not match real output:\n{prompt}",
        )

    def test_persona_precedence_claim(self):
        """Doc says `personas` wins over `persona` when both are set."""
        import inspect
        from terrarium.environments.dcops.is_this_seat_taken.is_this_seat_taken_env import (
            IsThisSeatTakenEnvironment,
        )
        src = inspect.getsource(IsThisSeatTakenEnvironment._resolve_personas)
        self.assertLess(src.index("if persona_map"), src.index("if persona_name"))


class DocumentedCLI(unittest.TestCase):
    def test_documented_flags_exist(self):
        main = read("examples/base_main.py")
        flags = set()
        for doc in (ENV_DOC, COMPACTION_DOC, PERSONAS_DOC):
            flags |= set(re.findall(r"(--[a-z][a-z-]+)", read(doc)))
        known = set(re.findall(r'add_argument\("(--[a-z-]+)"', main))
        # flags belonging to the harness, not base_main
        others = set()
        for entry in ("examples/acon_harness.py",
                      "terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_gui.py"):
            others |= set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', read(entry)))
        missing = sorted(f for f in flags if f not in known | others)
        self.assertEqual(missing, [], f"docs show flags nothing defines: {missing}")


class ShippedConfigs(unittest.TestCase):
    def test_no_dead_keys_in_seating_configs(self):
        import yaml
        blob = ""
        for root, _dirs, files in os.walk(os.path.join(REPO, "terrarium")):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    blob += read(os.path.relpath(os.path.join(root, f), REPO))
        blob += read("examples/base_main.py")

        cfg = yaml.safe_load(read("examples/configs/is_this_seat_taken.yaml"))
        dead = [k for k in cfg["environment"] if f'"{k}"' not in blob]
        self.assertEqual(dead, [], f"shipped config has keys nothing reads: {dead}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
