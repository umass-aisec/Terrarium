"""Regression tests: the Resource_Provisioner must see blackboard content.

`HospitalPrompts.get_user_prompt` used to return early for the provisioner
before building `bb_str`, so the blackboard section never reached its prompt
even though the provisioner instructions say "Check the Blackboard ...".
"""

import unittest

from terrarium.environments.dcops.hospital.hospital_prompts import HospitalPrompts


def _provisioner_context(phase="planning"):
    return {
        "phase": phase,
        "iteration": 1,
        "inventory_snapshot": {"Hospital_A": {"Ventilator": 3}},
        "failures_by_hospital": {"Hospital_A": {"Ventilator": 2}},
        "cost_table": [],
    }


def _department_context(phase="planning"):
    return {
        "phase": phase,
        "iteration": 1,
        "dept_info": {
            "capacity": 2,
            "service": "Triage",
            "hospital": "Hospital_A",
            "local_inventory": {"Ventilator": 1},
            "procedure_costs": {"Ventilator": 2},
        },
        "job_queue": [],
        "cost_table": [],
    }


BLACKBOARD = {
    "0": "[Triage_A] URGENT: Need Ventilator at Hospital_A\n",
    "7": "[Coalition] DEFICIT Oxygen_Tanks at Hospital_B\n",
}


class ProvisionerBlackboardVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.prompts = HospitalPrompts(env=None, config={})

    def test_provisioner_planning_prompt_includes_blackboard(self):
        prompt = self.prompts.get_user_prompt(
            "Resource_Provisioner", _provisioner_context("planning"), BLACKBOARD
        )
        self.assertIn("BLACKBOARD", prompt)
        self.assertIn("URGENT: Need Ventilator at Hospital_A", prompt)
        # Coalition-only board content must reach the provisioner too.
        self.assertIn("DEFICIT Oxygen_Tanks at Hospital_B", prompt)

    def test_provisioner_execution_prompt_includes_blackboard(self):
        prompt = self.prompts.get_user_prompt(
            "Resource_Provisioner", _provisioner_context("execution"), BLACKBOARD
        )
        self.assertIn("BLACKBOARD", prompt)
        self.assertIn("URGENT: Need Ventilator at Hospital_A", prompt)

    def test_provisioner_prompt_with_empty_blackboard(self):
        prompt = self.prompts.get_user_prompt(
            "Resource_Provisioner", _provisioner_context("planning"), {}
        )
        self.assertIn("BLACKBOARD", prompt)
        self.assertIn("(No new messages)", prompt)

    def test_department_prompt_still_includes_blackboard(self):
        prompt = self.prompts.get_user_prompt(
            "Triage_A", _department_context("planning"), BLACKBOARD
        )
        self.assertIn("BLACKBOARD", prompt)
        self.assertIn("URGENT: Need Ventilator at Hospital_A", prompt)


if __name__ == "__main__":
    unittest.main()
