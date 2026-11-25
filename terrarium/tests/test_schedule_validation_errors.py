"""Branch-local regression tests for fix/schedule_validation_errors.

Pins the invariant added on this branch: an invalid `schedule_patient` request
must surface as a failure to the agent *without* ending its turn, so the agent
can retry. Concretely, for each validation failure:

  - `HospitalEnvironment._process_schedule_request` raises
    `ScheduleRequestError` (a `ValueError` subclass) *before* mutating any
    state, and
  - `HospitalTools.execute_action` maps that exception to
    `{"status": "failed", "reason": ...}` **without** `action_committed`,
    which is the contract `BaseAgent` uses to decide whether the turn ends
    (see terrarium/agents/base.py: only a result carrying
    `action_committed: True` with a non-failed status commits the turn).

One test class per criterion, each with the same four checks: direct raise,
tool-level failure without commit, no state mutation, and a successful retry.
Covered so far: unknown patient_id; step_index beyond the pathway (including
negative indices); agent does not offer the required service; skipped step
(step N before step N-1); start_time below the minimum start; department at
full capacity at time t; booking reaching t >= max_time_horizon. Also pins
the deliberate exclusion: missing resources still commit (booked + -300/item
penalty), they are NOT a validation error.

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


class TestStepIndexBeyondPathway(unittest.TestCase):
    """Case 2: step_index beyond the length of the patient's pathway."""

    def test_process_schedule_request_raises(self):
        env, _ = _make_env()
        p_id, patient = _first_triage_patient(env)
        n = len(patient["pathway"])
        for bad_step in (n, n + 5, -1):
            with self.assertRaises(ScheduleRequestError, msg=f"step_index={bad_step}"):
                env._process_schedule_request(
                    "General_Hospital_Triage",
                    {"patient_id": p_id, "step_index": bad_step, "start_time": 0},
                )

    def test_tool_call_fails_without_committing(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        result = _schedule(tools, "General_Hospital_Triage", p_id, len(patient["pathway"]), 0)

        self.assertEqual(result.get("status"), "failed")
        self.assertIn(str(len(patient["pathway"])), result.get("reason", ""))
        # No `action_committed` on a failed call: BaseAgent keeps the turn.
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)

        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, p_id, len(patient["pathway"]) + 3, 0)

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)
        start = int(patient["arrival_time"])

        failed = _schedule(tools, agent, p_id, len(patient["pathway"]), start)
        self.assertEqual(failed.get("status"), "failed")
        self.assertNotIn("action_committed", failed)

        retried = _schedule(tools, agent, p_id, 0, start)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertIn(0, env.patient_states[p_id]["scheduled_steps"])


class TestServiceMismatch(unittest.TestCase):
    """Case 3: agent does not offer the service needed for that step."""

    def test_process_schedule_request_raises(self):
        env, _ = _make_env()
        p_id, _ = _first_triage_patient(env)
        # Step 0 of every pathway is Triage; a Surgery unit must be rejected.
        with self.assertRaises(ScheduleRequestError) as ctx:
            env._process_schedule_request(
                "General_Hospital_Surgery",
                {"patient_id": p_id, "step_index": 0, "start_time": 0},
            )
        self.assertIn("Surgery", str(ctx.exception))
        self.assertIn("Triage", str(ctx.exception))

    def test_tool_call_fails_without_committing(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        result = _schedule(tools, "General_Hospital_Surgery", p_id, 0, int(patient["arrival_time"]))

        self.assertEqual(result.get("status"), "failed")
        self.assertIn("Surgery", result.get("reason", ""))
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env()
        agent = "General_Hospital_Surgery"
        p_id, patient = _first_triage_patient(env)

        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, p_id, 0, int(patient["arrival_time"]))

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        start = int(patient["arrival_time"])

        failed = _schedule(tools, "General_Hospital_Surgery", p_id, 0, start)
        self.assertEqual(failed.get("status"), "failed")

        retried = _schedule(tools, "General_Hospital_Triage", p_id, 0, start)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertEqual(
            env.patient_states[p_id]["scheduled_steps"][0]["agent"],
            "General_Hospital_Triage",
        )


class TestSkippedStep(unittest.TestCase):
    """Case 4: scheduling step N while step N-1 is not yet scheduled."""

    def _step1_agent(self, env, patient):
        step1_service = patient["pathway"][1]["service"]
        return f"General_Hospital_{step1_service}"

    def test_process_schedule_request_raises(self):
        env, _ = _make_env()
        p_id, patient = _first_triage_patient(env)
        agent = self._step1_agent(env, patient)
        with self.assertRaises(ScheduleRequestError) as ctx:
            env._process_schedule_request(
                agent, {"patient_id": p_id, "step_index": 1, "start_time": 0}
            )
        self.assertIn("step 0", str(ctx.exception))

    def test_tool_call_fails_without_committing(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        agent = self._step1_agent(env, patient)
        result = _schedule(tools, agent, p_id, 1, int(patient["arrival_time"]))

        self.assertEqual(result.get("status"), "failed")
        self.assertIn("step 0", result.get("reason", ""))
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        agent = self._step1_agent(env, patient)

        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, p_id, 1, int(patient["arrival_time"]))

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        start0 = int(patient["arrival_time"])
        step1 = patient["pathway"][1]
        agent1 = self._step1_agent(env, patient)

        failed = _schedule(tools, agent1, p_id, 1, start0)
        self.assertEqual(failed.get("status"), "failed")
        self.assertNotIn("action_committed", failed)

        # Retry properly: book step 0 first, then step 1 after it ends.
        first = _schedule(tools, "General_Hospital_Triage", p_id, 0, start0)
        self.assertEqual(first.get("status"), "success")
        start1 = start0 + patient["pathway"][0]["duration"]
        retried = _schedule(tools, agent1, p_id, 1, start1)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertIn(1, env.patient_states[p_id]["scheduled_steps"])


class TestStartBelowMinimum(unittest.TestCase):
    """Case 5: start_time below the minimum start (arrival / prev end + penalty)."""

    def test_process_schedule_request_raises(self):
        env, _ = _make_env()
        p_id, patient = _first_triage_patient(env)
        arrival = int(patient["arrival_time"])
        with self.assertRaises(ScheduleRequestError) as ctx:
            env._process_schedule_request(
                "General_Hospital_Triage",
                {"patient_id": p_id, "step_index": 0, "start_time": arrival - 1},
            )
        self.assertIn(str(arrival), str(ctx.exception))

    def test_tool_call_fails_without_committing(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        arrival = int(patient["arrival_time"])
        result = _schedule(tools, "General_Hospital_Triage", p_id, 0, arrival - 1)

        self.assertEqual(result.get("status"), "failed")
        self.assertIn(str(arrival), result.get("reason", ""))
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)

        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, p_id, 0, int(patient["arrival_time"]) - 1)

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)
        arrival = int(patient["arrival_time"])

        failed = _schedule(tools, agent, p_id, 0, arrival - 1)
        self.assertEqual(failed.get("status"), "failed")
        self.assertNotIn("action_committed", failed)

        retried = _schedule(tools, agent, p_id, 0, arrival)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertIn(p_id, env.schedule[agent].get(arrival, []))

    def test_step_after_previous_end_is_enforced(self):
        """The minimum also covers the previous step's end time (same hospital)."""
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        arrival = int(patient["arrival_time"])
        dur0 = patient["pathway"][0]["duration"]
        step1_service = patient["pathway"][1]["service"]
        agent1 = f"General_Hospital_{step1_service}"

        first = _schedule(tools, "General_Hospital_Triage", p_id, 0, arrival)
        self.assertEqual(first.get("status"), "success")

        # One hour before step 0 ends is still below the minimum for step 1.
        too_early = arrival + dur0 - 1
        result = _schedule(tools, agent1, p_id, 1, too_early)
        self.assertEqual(result.get("status"), "failed")
        self.assertNotIn("action_committed", result)

        ok = _schedule(tools, agent1, p_id, 1, arrival + dur0)
        self.assertEqual(ok.get("status"), "success")


class TestCapacityFull(unittest.TestCase):
    """Case 6: the department is already at full capacity at time t."""

    # Triage has capacity 4 and duration 1, so a 5th booking at the same
    # time slot on one Triage unit must be rejected.
    def _fill_triage(self, env, tools, agent, p_ids, t):
        for p_id in p_ids:
            result = _schedule(tools, agent, p_id, 0, t)
            self.assertEqual(result.get("status"), "success", f"setup booking {p_id}")

    def test_process_schedule_request_raises(self):
        env, tools = _make_env(seed=7, num_patients=6)
        agent = "General_Hospital_Triage"
        p_ids = list(env.patients.keys())
        t = max(int(p["arrival_time"]) for p in env.patients.values())

        self._fill_triage(env, tools, agent, p_ids[:4], t)
        with self.assertRaises(ScheduleRequestError) as ctx:
            env._process_schedule_request(
                agent, {"patient_id": p_ids[4], "step_index": 0, "start_time": t}
            )
        self.assertIn(str(t), str(ctx.exception))

    def test_tool_call_fails_without_committing(self):
        env, tools = _make_env(seed=7, num_patients=6)
        agent = "General_Hospital_Triage"
        p_ids = list(env.patients.keys())
        t = max(int(p["arrival_time"]) for p in env.patients.values())

        self._fill_triage(env, tools, agent, p_ids[:4], t)
        result = _schedule(tools, agent, p_ids[4], 0, t)

        self.assertEqual(result.get("status"), "failed")
        self.assertIn(str(t), result.get("reason", ""))
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env(seed=7, num_patients=6)
        agent = "General_Hospital_Triage"
        p_ids = list(env.patients.keys())
        t = max(int(p["arrival_time"]) for p in env.patients.values())

        self._fill_triage(env, tools, agent, p_ids[:4], t)
        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        failures_before = dict(env.resource_failures)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, p_ids[4], 0, t)

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.resource_failures, failures_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        env, tools = _make_env(seed=7, num_patients=6)
        agent = "General_Hospital_Triage"
        p_ids = list(env.patients.keys())
        t = max(int(p["arrival_time"]) for p in env.patients.values())

        self._fill_triage(env, tools, agent, p_ids[:4], t)
        failed = _schedule(tools, agent, p_ids[4], 0, t)
        self.assertEqual(failed.get("status"), "failed")
        self.assertNotIn("action_committed", failed)

        # Same patient, one hour later: capacity is free again, commits.
        retried = _schedule(tools, agent, p_ids[4], 0, t + 1)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertIn(p_ids[4], env.schedule[agent].get(t + 1, []))


class TestMissingResourcesStillCommits(unittest.TestCase):
    """Exclusion pinned by the spec: lack of resources is NOT a failure.

    A schedule_patient that passes all validation checks but finds no
    inventory must still commit: the step is booked, the tool reports
    success with action_committed, and a -300/item penalty is recorded.
    """

    def test_missing_resources_commits_and_penalizes(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        hospital = "General_Hospital"
        p_id, patient = _first_triage_patient(env)
        start = int(patient["arrival_time"])

        # Drain exactly what Triage consumes so the booking finds nothing.
        costs = env.consumption_map["Triage"]
        for res in costs:
            env.inventory[hospital][res] = 0
        failures_before = dict(env.resource_failures)

        result = _schedule(tools, agent, p_id, 0, start)

        self.assertEqual(result.get("status"), "success")
        self.assertTrue(result.get("action_committed"))
        self.assertEqual(env.patient_states[p_id]["scheduled_steps"][0]["agent"], agent)
        self.assertIn(p_id, env.schedule[agent].get(start, []))

        # Penalty recorded: one failure per missing resource type, and the
        # patient-level failure log mentions the step.
        missing = list(costs.keys())
        for res in missing:
            self.assertEqual(
                env.resource_failures[res],
                failures_before[res] + 1,
                f"expected one recorded failure for {res}",
            )
        failures = env.patient_states[p_id]["resource_failures"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["step"], 0)
        self.assertEqual(sorted(failures[0]["missing"]), sorted(missing))


class TestBeyondTimeHorizon(unittest.TestCase):
    """Case 7: the booking would reach t >= max_time_horizon."""

    def test_process_schedule_request_raises(self):
        env, _ = _make_env()
        p_id, patient = _first_triage_patient(env)
        agent = "General_Hospital_Triage"
        for bad_start in (env.max_time_horizon, env.max_time_horizon + 50):
            with self.assertRaises(ScheduleRequestError, msg=f"start={bad_start}"):
                env._process_schedule_request(
                    agent, {"patient_id": p_id, "step_index": 0, "start_time": bad_start}
                )

    def test_tool_call_fails_without_committing(self):
        env, tools = _make_env()
        p_id, patient = _first_triage_patient(env)
        result = _schedule(tools, "General_Hospital_Triage", p_id, 0, env.max_time_horizon)

        self.assertEqual(result.get("status"), "failed")
        self.assertIn(str(env.max_time_horizon), result.get("reason", ""))
        self.assertNotIn("action_committed", result)

    def test_rejected_call_mutates_no_state(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)

        schedule_before = copy.deepcopy(env.schedule[agent])
        inventory_before = copy.deepcopy(env.inventory)
        patient_states_before = copy.deepcopy(env.patient_states)
        reward_before = env.get_final_summary()["joint_reward"]

        result = _schedule(tools, agent, p_id, 0, env.max_time_horizon)

        self.assertEqual(result.get("status"), "failed")
        self.assertEqual(env.schedule[agent], schedule_before)
        self.assertEqual(env.inventory, inventory_before)
        self.assertEqual(env.patient_states, patient_states_before)
        self.assertEqual(env.get_final_summary()["joint_reward"], reward_before)

    def test_agent_can_retry_after_failure(self):
        env, tools = _make_env()
        agent = "General_Hospital_Triage"
        p_id, patient = _first_triage_patient(env)

        failed = _schedule(tools, agent, p_id, 0, env.max_time_horizon)
        self.assertEqual(failed.get("status"), "failed")
        self.assertNotIn("action_committed", failed)

        # One hour earlier fits inside the horizon and commits.
        retried = _schedule(tools, agent, p_id, 0, env.max_time_horizon - 1)
        self.assertEqual(retried.get("status"), "success")
        self.assertTrue(retried.get("action_committed"))
        self.assertIn(p_id, env.schedule[agent].get(env.max_time_horizon - 1, []))

    def test_duration_spanning_past_horizon_is_rejected(self):
        """start < horizon but start + duration crosses it (e.g. a 24h Ward)."""
        env, tools = _make_env(seed=7, num_patients=8)
        # seed 7: Patient_0 has pathway [Triage(1), Ward(24)], arrival 4.
        p_id = "Patient_0"
        patient = env.patients[p_id]
        self.assertEqual(patient["pathway"][1]["service"], "Ward")
        self.assertGreater(patient["pathway"][1]["duration"], 1)

        start0 = int(patient["arrival_time"])
        first = _schedule(tools, "General_Hospital_Triage", p_id, 0, start0)
        self.assertEqual(first.get("status"), "success")

        # start = horizon - 1 passes the min-start check but the 24h stay
        # reaches t = horizon and beyond.
        ward_agent = "General_Hospital_Ward"
        start1 = env.max_time_horizon - 1
        result = _schedule(tools, ward_agent, p_id, 1, start1)
        self.assertEqual(result.get("status"), "failed")
        self.assertIn(str(env.max_time_horizon), result.get("reason", ""))
        self.assertNotIn("action_committed", result)
        self.assertNotIn(1, env.patient_states[p_id]["scheduled_steps"])


if __name__ == "__main__":
    unittest.main()
