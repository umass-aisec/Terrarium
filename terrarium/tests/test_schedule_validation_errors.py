"""Branch-local regression tests for fix/schedule_validation_errors.

Pins the invariant added on this branch: an invalid `schedule_patient` request
must surface as a failure to the agent *without* ending its turn, so the agent
can retry. Concretely, for each validation failure (currently: unknown
patient_id — more cases to be added on this branch):

  - `HospitalEnvironment._process_schedule_request` raises
    `ScheduleRequestError` (a `ValueError` subclass) *before* mutating any
    state, and
  - `HospitalTools.execute_action` maps that exception to
    `{"status": "failed", "reason": ...}` **without** `action_committed`,
    which is the contract `BaseAgent` uses to decide whether the turn ends
    (see terrarium/agents/base.py: only a result carrying
    `action_committed: True` with a non-failed status commits the turn).

Missing resources is intentionally NOT covered here: that case still commits
(the step is booked and a -300/item penalty is recorded), it must not become
an error.

Runs without any LLM/API access: the environment is instantiated directly with
a stub protocol (same pattern as test_collusion_runs_per_seed.py) and driven
through the real tool entry point (HospitalTools.handle_tool_call).

Run from the repo root with the local package shadowing any installed copy:

    PYTHONPATH=$PWD <python> -m unittest terrarium.tests.test_schedule_validation_errors -v
"""

import copy
import unittest
from typing import Any, Dict

from terrarium.environments.dcops.hospital.hospital_env import (
    HospitalEnvironment,
    ScheduleRequestError,
)
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


def _first_triage_patient(env):
    """First patient whose step 0 is Triage (all templates start with Triage)."""
    for p_id, patient in env.patients.items():
        if patient["pathway"][0]["service"] == "Triage":
            return p_id, patient
    raise AssertionError("no Triage-first patient generated")


class TestUnknownPatientId(unittest.TestCase):
    """Case 1: schedule_patient targets an unknown/invalid patient ID."""

    def test_process_schedule_request_raises(self):
        env, _ = _make_env()
        with self.assertRaises(ScheduleRequestError) as ctx:
            env._process_schedule_request(
                "General_Hospital_Triage",
                {"patient_id": "Patient_999", "step_index": 0, "start_time": 0},
            )
        # ScheduleRequestError must remain a ValueError subclass.
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertIn("Patient_999", str(ctx.exception))

    def test_tool_call_fails_without_committing(self):
        """The failure reaches the agent, and the turn is NOT consumed."""
        env, tools = _make_env()
        result = _schedule(tools, "General_Hospital_Triage", "Patient_999", 0, 0)

        self.assertEqual(result.get("status"), "failed")
        self.assertIn("Patient_999", result.get("reason", ""))
        # Contract with BaseAgent: no `action_committed` on a failed call, so
        # `_env_state_committed` stays False and the agent keeps its turn.
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"

        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        failures_before = dict(env.resource_failures)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, "Not_A_Patient", 0, 5)

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.resource_failures, failures_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        """Positive control: after a rejected call, a valid retry commits."""
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)
        start = int(patient["arrival_time"])

        failed = _schedule(tools, agent, "Patient_999", 0, start)
        self.assertEqual(failed.get("status"), "failed")
        self.assertNotIn("action_committed", failed)

        retried = _schedule(tools, agent, p_id, 0, start)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertEqual(env.patient_states[p_id]["scheduled_steps"][0]["agent"], agent)
        self.assertIn(p_id, env.schedule[agent].get(start, []))


if __name__ == "__main__":
    unittest.main()
