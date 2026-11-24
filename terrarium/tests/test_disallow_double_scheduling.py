"""Branch-local regression tests for fix/disallow_double_scheduling.

Pins the invariant added on this branch: a second `schedule_patient` call for
an already-scheduled (patient, step_index) pair must be a no-op — no resource
consumption, no schedule entry on the second agent, no change to
`scheduled_steps` ownership, and no change to the joint reward.

Runs without any LLM/API access: the environment is instantiated directly with
a stub protocol (same pattern as test_collusion_runs_per_seed.py) and driven
through the real tool entry point (HospitalTools.handle_tool_call).

Run from the repo root with the local package shadowing any installed copy:

    PYTHONPATH=$PWD <python> -m unittest terrarium.tests.test_disallow_double_scheduling -v
    # or, if pytest is available:
    PYTHONPATH=$PWD <python> -m pytest terrarium/tests/test_disallow_double_scheduling.py -v
"""

import copy
import unittest
from typing import Any, Dict

from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment
from terrarium.environments.dcops.hospital.hospital_tools import HospitalTools


class _ProtocolStub:
    environment = None


def _make_env(seed: int = 7, num_patients: int = 3):
    env = HospitalEnvironment(
        communication_protocol=_ProtocolStub(),
        config={
            "simulation": {"seed": seed},
            "environment": {"num_hospitals": 2, "num_patients": num_patients},
        },
        tool_logger=None,
    )
    tools = HospitalTools(blackboard_manager=None, environment=env)
    return env, tools


def _schedule(tools, agent, patient_id, step_index, start_time) -> Dict[str, Any]:
    return tools.handle_tool_call(
        "schedule_patient",
        agent,
        {
            "patient_id": patient_id,
            "step_index": step_index,
            "start_time": start_time,
            "rationale": "test",
        },
        phase="execution",
        iteration=1,
    )


class TestDisallowDoubleScheduling(unittest.TestCase):
    def _triage_patient(self, env):
        """First patient whose step 0 is Triage (all templates start with Triage)."""
        for p_id, patient in env.patients.items():
            if patient["pathway"][0]["service"] == "Triage":
                return p_id, patient
        self.fail("no Triage-first patient generated")

    def test_second_schedule_same_patient_same_step_is_noop(self):
        env, tools = _make_env(seed=7, num_patients=3)
        p_id, patient = self._triage_patient(env)
        start = int(patient["arrival_time"])

        agent_a = "General_Hospital_Triage"
        agent_b = "St_Marys_Center_Triage"
        hospital_b = "St_Marys_Center"

        first = _schedule(tools, agent_a, p_id, 0, start)
        self.assertEqual(first["status"], "success")
        self.assertEqual(env.patient_states[p_id]["scheduled_steps"][0]["agent"], agent_a)
        self.assertIn(p_id, env.schedule[agent_a].get(start, []))

        # Snapshot everything the double-booking bug used to corrupt.
        inventory_b_before = copy.deepcopy(env.inventory[hospital_b])
        failures_before = dict(env.resource_failures)
        schedule_b_before = copy.deepcopy(env.schedule[agent_b])
        reward_before = env.get_final_summary()["joint_reward"]

        # The attack: a different unit tries to book the same patient at the same step.
        second = _schedule(tools, agent_b, p_id, 0, start)

        self.assertEqual(
            second["status"],
            "failed",
            "second schedule_patient call must be rejected",
        )
        self.assertEqual(
            env.patient_states[p_id]["scheduled_steps"][0]["agent"],
            agent_a,
            "scheduled_steps ownership must not move to the second agent",
        )
        self.assertEqual(
            env.schedule[agent_b],
            schedule_b_before,
            "second agent must not gain a schedule entry",
        )
        self.assertEqual(
            env.inventory[hospital_b],
            inventory_b_before,
            "second agent's hospital inventory must not be consumed",
        )
        self.assertEqual(
            env.resource_failures,
            failures_before,
            "no new resource failures may be recorded",
        )
        self.assertEqual(
            env.get_final_summary()["joint_reward"],
            reward_before,
            "joint reward must be unchanged by the rejected double-booking",
        )
        # Sanity: the second call still returns a well-formed result (the current
        # implementation exits silently rather than raising).
        self.assertIn(second.get("status"), ("success", "failed"))

    def test_guard_is_scoped_to_patient_and_step(self):
        """Positive control: scheduling a *different* patient (or step) still works."""
        env, tools = _make_env(seed=7, num_patients=3)
        agent_b = "St_Marys_Center_Triage"

        p_ids = [p for p, pat in env.patients.items() if pat["pathway"][0]["service"] == "Triage"]
        self.assertGreaterEqual(len(p_ids), 2, "need at least two Triage-first patients for this control")

        p1, p2 = p_ids[0], p_ids[1]
        start1 = int(env.patients[p1]["arrival_time"])
        start2 = int(env.patients[p2]["arrival_time"])

        _schedule(tools, "General_Hospital_Triage", p1, 0, start1)
        result = _schedule(tools, agent_b, p2, 0, start2)

        self.assertEqual(result["status"], "success")
        self.assertEqual(env.patient_states[p2]["scheduled_steps"][0]["agent"], agent_b)
        self.assertIn(p2, env.schedule[agent_b].get(start2, []))


if __name__ == "__main__":
    unittest.main()
