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


class DocumentedCLI(unittest.TestCase):
    def test_documented_flags_exist(self):
        flags = set()
        for doc in COMPONENT_DOCS:
            flags |= set(re.findall(r"(--[a-z][a-z-]+)", read(doc)))
        self.assertGreaterEqual(len(flags), 4, "expected the docs to show CLI flags")

        known = set()
        for entry in (
            "examples/base_main.py",
            "examples/acon_harness.py",
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
