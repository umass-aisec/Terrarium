import unittest

from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment
from terrarium.personas import PRESETS, build_persona_prompt, build_trait_clause


class _ProtocolStub:
    environment = None


def _make_env(env_config=None):
    config = {
        "simulation": {"seed": 7},
        "environment": {"num_hospitals": 1, "num_patients": 2, **(env_config or {})},
    }
    return HospitalEnvironment(
        communication_protocol=_ProtocolStub(),
        config=config,
        tool_logger=None,
    )


def _full_persona_map(env, **overrides):
    personas = {name: "diplomat" for name in env.agent_names}
    personas.update(overrides)
    return personas


class ResolvePersonasTests(unittest.TestCase):
    def test_no_config_gives_none_for_every_agent(self):
        env = _make_env()
        self.assertEqual(
            env.persona_by_agent,
            {name: None for name in env.agent_names},
        )

    def test_uniform_persona_applies_to_every_agent(self):
        env = _make_env({"persona": "diplomat"})
        self.assertEqual(set(env.persona_by_agent), set(env.agent_names))
        for name in env.agent_names:
            self.assertIs(env.persona_by_agent[name], PRESETS["diplomat"])

    def test_per_agent_map_resolves_each_agent(self):
        base = _make_env()
        env = _make_env(
            {"personas": _full_persona_map(base, **{"General_Hospital_Surgery": "territorial"})}
        )
        self.assertIs(env.persona_by_agent["General_Hospital_Surgery"], PRESETS["territorial"])
        self.assertIs(env.persona_by_agent["Resource_Provisioner"], PRESETS["diplomat"])

    def test_unknown_uniform_preset_raises_with_available_list(self):
        with self.assertRaises(ValueError) as ctx:
            _make_env({"persona": "not a preset"})
        self.assertIn("Available presets", str(ctx.exception))

    def test_unknown_preset_in_map_raises(self):
        base = _make_env()
        with self.assertRaises(ValueError) as ctx:
            _make_env({"personas": _full_persona_map(base, **{"General_Hospital_Ward": "bogus"})})
        self.assertIn("bogus", str(ctx.exception))

    def test_missing_agent_entry_raises(self):
        base = _make_env()
        personas = _full_persona_map(base)
        removed = personas.pop("General_Hospital_Ward")
        self.assertIsNotNone(removed)
        with self.assertRaises(ValueError) as ctx:
            _make_env({"personas": personas})
        self.assertIn("General_Hospital_Ward", str(ctx.exception))


class SystemPromptPersonaTests(unittest.TestCase):
    def test_persona_wraps_base_prompt_with_trait_clause(self):
        base_env = _make_env()
        env = _make_env(
            {"personas": _full_persona_map(base_env, **{"General_Hospital_Surgery": "territorial"})}
        )
        baseline = base_env.prompts.get_system_prompt()
        prompt = env.prompts.get_system_prompt("General_Hospital_Surgery")

        self.assertIn("For the following task, respond in a way that matches this description", prompt)
        self.assertIn("A territorial holdout.", prompt)
        self.assertIn(build_trait_clause(PRESETS["territorial"]), prompt)
        self.assertIn("You are a Hospital Agent", prompt)
        self.assertEqual(prompt, build_persona_prompt(PRESETS["territorial"], baseline))

    def test_uniform_persona_wraps_every_agent_prompt(self):
        env = _make_env({"persona": "cautious skeptic"})
        for name in env.agent_names:
            prompt = env.prompts.get_system_prompt(name)
            self.assertIn("A cautious skeptic.", prompt)
            self.assertIn(build_trait_clause(PRESETS["cautious skeptic"]), prompt)

    def test_agent_without_persona_gets_baseline_unchanged(self):
        base_env = _make_env()
        env = _make_env(
            {"personas": _full_persona_map(base_env, **{"General_Hospital_Surgery": "territorial"})}
        )
        baseline = base_env.prompts.get_system_prompt()
        # A per-agent map must cover every agent, so only names outside the map
        # (and the no-arg default) fall through to the baseline prompt.
        self.assertEqual(env.prompts.get_system_prompt("Not_A_Configured_Agent"), baseline)
        self.assertEqual(env.prompts.get_system_prompt(""), baseline)

    def test_no_config_returns_baseline_unchanged(self):
        env = _make_env()
        baseline = _make_env().prompts.get_system_prompt()
        self.assertEqual(env.prompts.get_system_prompt("General_Hospital_Surgery"), baseline)

    def test_no_arg_call_still_works(self):
        env = _make_env({"persona": "diplomat"})
        baseline = _make_env().prompts.get_system_prompt()
        self.assertEqual(env.prompts.get_system_prompt(), baseline)


if __name__ == "__main__":
    unittest.main()
