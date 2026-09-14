import unittest

from terrarium.personas import (
    CONFIDENT_COLLABORATOR,
    Persona,
    build_persona_prompt,
    build_trait_clause,
    qualify,
)
from terrarium.personas.adjectives import get_domain_adjectives, get_facet_adjectives


class QualifyTests(unittest.TestCase):
    def test_neutral_level_contributes_nothing(self):
        self.assertIsNone(qualify(5, "introverted", "extraverted"))

    def test_low_levels_use_low_adjective(self):
        self.assertEqual(qualify(1, "introverted", "extraverted"), "extremely introverted")
        self.assertEqual(qualify(2, "introverted", "extraverted"), "very introverted")
        self.assertEqual(qualify(3, "introverted", "extraverted"), "introverted")
        self.assertEqual(qualify(4, "introverted", "extraverted"), "a bit introverted")

    def test_high_levels_use_high_adjective(self):
        self.assertEqual(qualify(6, "introverted", "extraverted"), "a bit extraverted")
        self.assertEqual(qualify(7, "introverted", "extraverted"), "extraverted")
        self.assertEqual(qualify(8, "introverted", "extraverted"), "very extraverted")
        self.assertEqual(qualify(9, "introverted", "extraverted"), "extremely extraverted")

    def test_out_of_range_level_raises(self):
        with self.assertRaises(ValueError):
            qualify(0, "introverted", "extraverted")
        with self.assertRaises(ValueError):
            qualify(10, "introverted", "extraverted")


class AdjectiveLookupTests(unittest.TestCase):
    def test_domain_lookup_is_case_insensitive(self):
        self.assertEqual(get_domain_adjectives("ext"), get_domain_adjectives("EXT"))

    def test_facet_lookup_by_code_or_name(self):
        self.assertEqual(get_facet_adjectives("A1"), get_facet_adjectives("Trust"))
        self.assertEqual(get_facet_adjectives("A1"), [("distrustful", "trustful")])

    def test_unknown_domain_raises(self):
        with self.assertRaises(ValueError):
            get_domain_adjectives("XYZ")

    def test_unknown_facet_raises(self):
        with self.assertRaises(ValueError):
            get_facet_adjectives("not-a-facet")


class TraitClauseTests(unittest.TestCase):
    def test_all_neutral_yields_empty_clause(self):
        persona = Persona.from_big_five()
        self.assertEqual(build_trait_clause(persona), "")

    def test_domain_level_defaults_to_compact(self):
        persona = Persona(traits={"EXT": 7})
        clause = build_trait_clause(persona)
        self.assertTrue(clause.startswith("I'm "))
        self.assertIn("talkative", clause)
        self.assertIn("bold", clause)
        self.assertIn("extraverted", clause)
        # compact EXT is 4 pairs; "friendly"/"cheerful" only appear in full mode.
        self.assertNotIn("friendly", clause)
        self.assertNotIn("cheerful", clause)

    def test_domain_level_full_covers_every_facet_in_domain(self):
        persona = Persona(traits={"EXT": 7}, verbosity="full")
        clause = build_trait_clause(persona)
        self.assertIn("friendly", clause)
        self.assertIn("cheerful", clause)
        self.assertIn("talkative", clause)

    def test_same_persona_renders_identically_across_calls(self):
        persona = Persona(traits={"EXT": 7, "NEU": 3})
        self.assertEqual(build_trait_clause(persona), build_trait_clause(persona))

    def test_invalid_verbosity_raises(self):
        with self.assertRaises(ValueError):
            Persona(traits={"EXT": 7}, verbosity="verbose")

    def test_facet_level_restricts_to_single_facet(self):
        persona = Persona(traits={"A1": 8})
        self.assertEqual(build_trait_clause(persona), "I'm very trustful.")

    def test_neutral_domains_are_skipped(self):
        # confident collaborator sets CON=5 (neutral); it should not appear.
        clause = build_trait_clause(CONFIDENT_COLLABORATOR)
        self.assertNotIn("hardworking", clause)
        self.assertNotIn("lazy", clause)


class PersonaPromptTests(unittest.TestCase):
    def test_prompt_wraps_task_with_description_and_trait_clause(self):
        persona = Persona(description="A helpful assistant.", traits={"EXT": 7})
        prompt = build_persona_prompt(persona, task="Summarize this document.")
        self.assertIn("A helpful assistant.", prompt)
        self.assertIn("I'm", prompt)
        self.assertTrue(prompt.endswith("Summarize this document."))

    def test_empty_persona_returns_task_unchanged(self):
        persona = Persona()
        self.assertEqual(build_persona_prompt(persona, task="Do the thing."), "Do the thing.")


if __name__ == "__main__":
    unittest.main()
