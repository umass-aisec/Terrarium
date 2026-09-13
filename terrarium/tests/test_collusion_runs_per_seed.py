import json
import tempfile
import unittest
from pathlib import Path

from terrarium.core.blackboard import Megaboard
from terrarium.environments.dcops.hospital.hospital_env import HospitalEnvironment

try:
    from experiments.collusion import resume as resume_mod
    from experiments.collusion import run as run_mod
    from experiments.collusion.plots import plot_sweep
    from experiments.collusion.plots.generate_regret_report import (
        RunRow,
        _condition_for_row as regret_condition_for_row,
        _condition_keys as regret_condition_keys,
        _load_run_row,
        _seed_means as regret_seed_means,
    )
    from experiments.collusion.compute_meeting_scheduling_optimal import (
        MeetingSchedulingInstanceData,
        MeetingSpec,
        VariableSpecLite,
        evaluate_assignment,
        solve_optimal_assignment,
    )
    from experiments.collusion.metrics import compute_collusion_metrics
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("experiments"):
        raise unittest.SkipTest("Optional experiments package is not available") from exc
    raise


class CollusionRunsPerSeedTests(unittest.TestCase):
    class _ProtocolStub:
        environment = None

    def _config(
        self,
        *,
        seeds=None,
        runs_per_seed=1,
        secret_flags=None,
        prompt_variants=None,
    ):
        return {
            "simulation": {
                "max_iterations": 1,
                "max_planning_rounds": 1,
                "max_conversation_steps": 1,
                "seed": 1,
            },
            "environment": {"name": "JiraTicketEnvironment"},
            "communication_network": {"topology": "complete", "num_agents": 6},
            "llm_models": [{"label": "model-a", "llm": {"provider": "openai"}}],
            "experiment": {
                "tag": "collusion_test",
                "seeds": list(seeds or [1]),
                "runs_per_seed": int(runs_per_seed),
                "sweeps": [
                    {
                        "name": "sweep-a",
                        "topologies": ["complete"],
                        "num_agents": [6],
                        "colluder_counts": [2],
                        "secret_channel_enabled": list(secret_flags or [False]),
                        "prompt_variants": list(prompt_variants or ["control"]),
                    }
                ],
            },
        }

    def _write_complete_run(
        self,
        *,
        root: Path,
        spec: run_mod.RunSpec,
        include_replica_index: bool,
    ) -> Path:
        run_dir = root / "runs" / spec.model_label / spec.sweep_name / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "run_id": spec.run_id,
            "model_label": spec.model_label,
            "provider": "openai",
            "model": "gpt-test",
            "sweep": spec.sweep_name,
            "topology": spec.topology,
            "num_agents": spec.num_agents,
            "colluder_count": spec.colluder_count,
            "secret_channel_enabled": spec.secret_channel_enabled,
            "prompt_variant": spec.effective_prompt_variant,
            "seed": spec.seed,
            "colluders": ["agent_0", "agent_1"],
        }
        if include_replica_index:
            run_config["replica_index"] = spec.replica_index

        artifacts = {
            "run_config.json": run_config,
            "metrics.json": {
                "status": "complete",
                "coalition_mean_regret": 0.1,
                "noncoalition_mean_regret": 0.2,
            },
            "final_summary.json": {"status": "complete", "joint_reward": 1.0},
            "agent_turns.json": [],
            "tool_events.json": [],
            "blackboards.json": [],
        }
        for filename, payload in artifacts.items():
            (run_dir / filename).write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return run_dir

    def test_iter_expected_run_specs_expands_runs_per_seed(self):
        cfg = self._config(
            seeds=[11, 12],
            runs_per_seed=2,
            secret_flags=[False],
            prompt_variants=["control", "simple"],
        )

        specs = list(run_mod._iter_expected_run_specs(cfg))

        self.assertEqual(len(specs), 4)
        self.assertEqual(
            [spec.run_id for spec in specs],
            [
                "model-a__sweep-a__complete__n6__c2__secret0__pvcontrol__seed11",
                "model-a__sweep-a__complete__n6__c2__secret0__pvcontrol__seed11__replica1",
                "model-a__sweep-a__complete__n6__c2__secret0__pvcontrol__seed12",
                "model-a__sweep-a__complete__n6__c2__secret0__pvcontrol__seed12__replica1",
            ],
        )
        self.assertTrue(all(spec.effective_prompt_variant == "control" for spec in specs))

    def test_iter_expected_run_specs_supports_secret_channel_counts(self):
        cfg = self._config(
            seeds=[11],
            runs_per_seed=1,
            secret_flags=[False],
            prompt_variants=["control"],
        )
        cfg["experiment"]["sweeps"][0].pop("secret_channel_enabled")
        cfg["experiment"]["sweeps"][0]["secret_channel_counts"] = [0, 1, 2, 3]

        specs = list(run_mod._iter_expected_run_specs(cfg))

        self.assertEqual([spec.secret_channel_count for spec in specs], [0, 1, 2, 3])
        self.assertEqual(
            [spec.run_id for spec in specs],
            [
                "model-a__sweep-a__complete__n6__c2__secret0__pvcontrol__seed11",
                "model-a__sweep-a__complete__n6__c2__secret1__pvcontrol__seed11",
                "model-a__sweep-a__complete__n6__c2__secret1__sc2__pvcontrol__seed11",
                "model-a__sweep-a__complete__n6__c2__secret1__sc3__pvcontrol__seed11",
            ],
        )

    def test_noncolluder_secret_pairs_are_disjoint_and_exclude_colluders(self):
        pairs = run_mod._select_noncolluder_secret_pairs(
            agent_names=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
            colluders=["A", "C", "E", "G"],
            secret_channel_count=3,
        )

        self.assertEqual(pairs, [["B", "D"], ["F", "H"]])
        paired_agents = [agent for pair in pairs for agent in pair]
        self.assertEqual(len(paired_agents), len(set(paired_agents)))
        self.assertTrue({"A", "C", "E", "G"}.isdisjoint(paired_agents))

    def test_secret_blackboard_can_duplicate_public_participants(self):
        board = Megaboard()
        public_id = board.add_blackboard(["A", "B"])
        secret_id = board.add_blackboard(
            ["A", "B"],
            {"secret_channel": True, "visibility": "secret", "allow_duplicate": True},
        )

        self.assertEqual(public_id, 0)
        self.assertEqual(secret_id, 1)
        self.assertEqual(len(board.blackboards), 2)
        self.assertFalse(board.blackboards[0].template.get("secret_channel", False))
        self.assertTrue(board.blackboards[1].template["secret_channel"])

    def test_iter_expected_run_specs_preserves_mixed_agent_profile(self):
        cfg = self._config(seeds=[1], secret_flags=[True], prompt_variants=["simple"])
        gpt_cfg = {"provider": "foundry", "foundry": {"model": "gpt-5.4"}}
        opus_cfg = {"provider": "foundry", "foundry": {"model": "claude-opus-4-6"}}
        cfg["llm_models"] = [
            {
                "label": "gpt54_opus46_colluders",
                "llm": gpt_cfg,
                "agent_llms": [gpt_cfg, gpt_cfg, opus_cfg, gpt_cfg, gpt_cfg, gpt_cfg],
                "collusion": {"colluders": {"indices": [0, 2]}},
            }
        ]

        specs = list(run_mod._iter_expected_run_specs(cfg))

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].model_label, "gpt54_opus46_colluders")
        self.assertEqual(specs[0].model_agent_llms[2]["foundry"]["model"], "claude-opus-4-6")
        self.assertEqual(
            specs[0].model_collusion_cfg["colluders"]["indices"],
            [0, 2],
        )

    def test_explicit_colluder_indices_resolve_from_agent_order(self):
        colluders = run_mod._resolve_colluders_from_config(
            agent_names=["Avery", "Jordan", "Quinn"],
            count=2,
            collusion_cfg={"colluders": {"indices": [0, 2]}},
            rng=run_mod.random.Random(1),
        )

        self.assertEqual(colluders, ["Avery", "Quinn"])

    def test_agent_llms_list_assigns_models_by_position(self):
        gpt_cfg = {"provider": "foundry", "foundry": {"model": "gpt-5.4"}}
        opus_cfg = {"provider": "foundry", "foundry": {"model": "claude-opus-4-6"}}

        resolved = run_mod._resolve_agent_llm_configs(
            agent_names=["Avery", "Jordan", "Quinn"],
            default_llm_cfg=gpt_cfg,
            assignment_cfg=[gpt_cfg, gpt_cfg, opus_cfg],
        )

        self.assertEqual(resolved["Avery"]["foundry"]["model"], "gpt-5.4")
        self.assertEqual(resolved["Jordan"]["foundry"]["model"], "gpt-5.4")
        self.assertEqual(resolved["Quinn"]["foundry"]["model"], "claude-opus-4-6")

    def test_resume_select_incomplete_runs_treats_legacy_run_as_replica_zero(self):
        cfg = self._config(seeds=[7], runs_per_seed=2)
        specs = list(run_mod._iter_expected_run_specs(cfg))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_complete_run(
                root=root, spec=specs[0], include_replica_index=False
            )

            incomplete, completed, total_runs, reason_counts, reasons_by_run_id = (
                resume_mod._select_incomplete_runs(
                    root=root,
                    cfg=cfg,
                    require_status_complete=True,
                    rerun_error_turns=True,
                )
            )

        self.assertEqual(total_runs, 2)
        self.assertEqual(completed, 1)
        self.assertEqual(len(incomplete), 1)
        self.assertEqual(incomplete[0].run_id, specs[1].run_id)
        self.assertEqual(reason_counts, {"missing_files": 1})
        self.assertEqual(reasons_by_run_id[specs[1].run_id], ["missing_files"])

    def test_rebuild_summary_files_infers_replica_index_for_legacy_and_new_runs(self):
        cfg = self._config(seeds=[5], runs_per_seed=2)
        specs = list(run_mod._iter_expected_run_specs(cfg))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_complete_run(
                root=root, spec=specs[0], include_replica_index=False
            )
            self._write_complete_run(
                root=root, spec=specs[1], include_replica_index=True
            )

            resume_mod._rebuild_summary_files(root)
            summary_rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        rows_by_id = {row["run_id"]: row for row in summary_rows}
        self.assertEqual(rows_by_id[specs[0].run_id]["replica_index"], 0)
        self.assertEqual(rows_by_id[specs[1].run_id]["replica_index"], 1)

    def test_plot_sweep_seed_means_average_replicas_within_seed(self):
        rows = [
            {"seed": 1, "joint_reward_ratio": 0.2},
            {"seed": 1, "joint_reward_ratio": 0.4},
            {"seed": 2, "joint_reward_ratio": 0.6},
        ]

        values = plot_sweep._seed_means(rows, "joint_reward_ratio")

        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 0.3)
        self.assertAlmostEqual(values[1], 0.6)

    def test_regret_report_seed_means_average_replicas_within_seed(self):
        rows = [
            RunRow(
                model_label="model-a",
                provider="openai",
                model="gpt-test",
                sweep_name="sweep-a",
                topology="complete",
                num_agents=6,
                colluder_count=2,
                seed=1,
                replica_index=0,
                secret_channel_enabled=False,
                prompt_variant="control",
                status="complete",
                joint_reward=0.0,
                optimal_joint_reward=1.0,
                normalized_regret=0.2,
                judge_mean_rating=None,
                coalition_mean_regret=0.1,
                noncoalition_mean_regret=0.2,
                coalition_advantage_mean=None,
            ),
            RunRow(
                model_label="model-a",
                provider="openai",
                model="gpt-test",
                sweep_name="sweep-a",
                topology="complete",
                num_agents=6,
                colluder_count=2,
                seed=1,
                replica_index=1,
                secret_channel_enabled=False,
                prompt_variant="control",
                status="complete",
                joint_reward=0.0,
                optimal_joint_reward=1.0,
                normalized_regret=0.4,
                judge_mean_rating=None,
                coalition_mean_regret=0.1,
                noncoalition_mean_regret=0.2,
                coalition_advantage_mean=None,
            ),
            RunRow(
                model_label="model-a",
                provider="openai",
                model="gpt-test",
                sweep_name="sweep-a",
                topology="complete",
                num_agents=6,
                colluder_count=2,
                seed=2,
                replica_index=0,
                secret_channel_enabled=False,
                prompt_variant="control",
                status="complete",
                joint_reward=0.0,
                optimal_joint_reward=1.0,
                normalized_regret=0.6,
                judge_mean_rating=None,
                coalition_mean_regret=0.1,
                noncoalition_mean_regret=0.2,
                coalition_advantage_mean=None,
            ),
        ]

        values = regret_seed_means(rows, key="normalized_regret")

        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 0.3)
        self.assertAlmostEqual(values[1], 0.6)

    def test_regret_report_uses_secret_channel_count_conditions(self):
        base_kwargs = {
            "model_label": "model-a",
            "provider": "openai",
            "model": "gpt-test",
            "sweep_name": "sweep-a",
            "topology": "complete",
            "num_agents": 9,
            "colluder_count": 4,
            "seed": 1,
            "replica_index": 0,
            "status": "partial_convergence",
            "joint_reward": 0.0,
            "optimal_joint_reward": None,
            "normalized_regret": 0.1,
            "judge_mean_rating": None,
            "coalition_mean_regret": 0.1,
            "noncoalition_mean_regret": 0.2,
            "coalition_advantage_mean": None,
        }
        rows = [
            RunRow(
                **base_kwargs,
                secret_channel_enabled=False,
                prompt_variant="control",
                secret_channel_count=0,
            ),
            RunRow(
                **base_kwargs,
                secret_channel_enabled=True,
                prompt_variant="control",
                secret_channel_count=1,
            ),
            RunRow(
                **base_kwargs,
                secret_channel_enabled=True,
                prompt_variant="control",
                secret_channel_count=2,
            ),
            RunRow(
                **base_kwargs,
                secret_channel_enabled=True,
                prompt_variant="control",
                secret_channel_count=3,
            ),
            RunRow(
                **base_kwargs,
                secret_channel_enabled=True,
                prompt_variant="simple",
                secret_channel_count=1,
            ),
        ]

        self.assertEqual(regret_condition_keys(rows), ["baseline", "sc1", "sc2", "sc3"])
        self.assertEqual(
            [
                regret_condition_for_row(row, channel_count_conditions=True)
                for row in rows
            ],
            ["baseline", "sc1", "sc2", "sc3", None],
        )

    def test_regret_report_uses_system_regret_ratio_without_optimal_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "model-a" / "sweep-a" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "model_label": "model-a",
                        "sweep": "sweep-a",
                        "topology": "complete",
                        "num_agents": 6,
                        "colluder_count": 2,
                        "secret_channel_enabled": False,
                        "prompt_variant": "control",
                        "seed": 1,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "final_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "joint_reward": 42.0,
                        "joint_reward_ratio": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "system_regret_ratio": 0.25,
                        "coalition_mean_regret": 0.0,
                        "noncoalition_mean_regret": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            row = _load_run_row(
                run_dir=run_dir,
                model_dir=run_dir.parents[1],
                sweep_name="sweep-a",
                compute_optimal=False,
                prefer_repaired=False,
            )

        self.assertIsNotNone(row)
        self.assertAlmostEqual(row.normalized_regret, 0.25)

    def test_hospital_final_summary_includes_agent_rewards(self):
        env = HospitalEnvironment(
            communication_protocol=self._ProtocolStub(),
            config={
                "simulation": {"seed": 7},
                "environment": {"num_hospitals": 2, "num_patients": 3},
            },
            tool_logger=None,
        )

        summary = env.get_final_summary()

        self.assertIn("agent_rewards", summary)
        self.assertEqual(set(summary["agent_rewards"].keys()), set(env.agent_names))
        self.assertAlmostEqual(
            sum(float(v) for v in summary["agent_rewards"].values()),
            float(summary["joint_reward"]),
        )

    def test_regret_report_load_run_row_uses_hospital_agent_rewards_from_final_summary(self):
        env = HospitalEnvironment(
            communication_protocol=self._ProtocolStub(),
            config={
                "simulation": {"seed": 11},
                "environment": {"num_hospitals": 2, "num_patients": 4},
            },
            tool_logger=None,
        )
        final_summary = env.get_final_summary()
        colluders = env.agent_names[:2]
        colluder_rewards = [float(final_summary["agent_rewards"][a]) for a in colluders]
        noncolluder_rewards = [
            float(final_summary["agent_rewards"][a])
            for a in env.agent_names
            if a not in colluders
        ]
        expected_advantage = (sum(colluder_rewards) / len(colluder_rewards)) - (
            sum(noncolluder_rewards) / len(noncolluder_rewards)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "model-a" / "sweep-a" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "model_label": "model-a",
                        "sweep": "sweep-a",
                        "environment_name": "HospitalChoiceEnvironment",
                        "environment_cfg": {
                            "import_path": "experiments.collusion.choice.hospital_choice_env:HospitalChoiceEnvironment",
                            "num_hospitals": 2,
                            "num_patients": 4,
                        },
                        "topology": "complete",
                        "num_agents": len(env.agent_names),
                        "colluder_count": len(colluders),
                        "secret_channel_enabled": False,
                        "prompt_variant": "control",
                        "seed": 11,
                        "colluders": colluders,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "final_summary.json").write_text(
                json.dumps(final_summary), encoding="utf-8"
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": final_summary["status"],
                        "coalition_advantage_mean": None,
                        "coalition_mean_regret": None,
                        "noncoalition_mean_regret": None,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "tool_events.json").write_text("[]", encoding="utf-8")

            row = _load_run_row(
                run_dir=run_dir,
                model_dir=run_dir.parents[1],
                sweep_name="sweep-a",
                compute_optimal=False,
                prefer_repaired=False,
            )

        self.assertIsNotNone(row)
        self.assertIsNotNone(row.coalition_advantage_mean)
        self.assertAlmostEqual(row.coalition_advantage_mean, expected_advantage)

    def test_regret_report_load_run_row_recovers_hospital_advantage_from_tool_events(self):
        env = HospitalEnvironment(
            communication_protocol=self._ProtocolStub(),
            config={
                "simulation": {"seed": 13},
                "environment": {"num_hospitals": 2, "num_patients": 4},
            },
            tool_logger=None,
        )
        final_summary = env.get_final_summary()
        colluders = env.agent_names[:2]
        colluder_rewards = [float(final_summary["agent_rewards"][a]) for a in colluders]
        noncolluder_rewards = [
            float(final_summary["agent_rewards"][a])
            for a in env.agent_names
            if a not in colluders
        ]
        expected_advantage = (sum(colluder_rewards) / len(colluder_rewards)) - (
            sum(noncolluder_rewards) / len(noncolluder_rewards)
        )
        legacy_summary = dict(final_summary)
        legacy_summary.pop("agent_rewards", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "model-a" / "sweep-a" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-a",
                        "model_label": "model-a",
                        "sweep": "sweep-a",
                        "environment_name": "HospitalChoiceEnvironment",
                        "environment_cfg": {
                            "import_path": "experiments.collusion.choice.hospital_choice_env:HospitalChoiceEnvironment",
                            "num_hospitals": 2,
                            "num_patients": 4,
                        },
                        "topology": "complete",
                        "num_agents": len(env.agent_names),
                        "colluder_count": len(colluders),
                        "secret_channel_enabled": False,
                        "prompt_variant": "control",
                        "seed": 13,
                        "colluders": colluders,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "final_summary.json").write_text(
                json.dumps(legacy_summary), encoding="utf-8"
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "status": legacy_summary["status"],
                        "coalition_advantage_mean": None,
                        "coalition_mean_regret": None,
                        "noncoalition_mean_regret": None,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "tool_events.json").write_text("[]", encoding="utf-8")

            row = _load_run_row(
                run_dir=run_dir,
                model_dir=run_dir.parents[1],
                sweep_name="sweep-a",
                compute_optimal=False,
                prefer_repaired=False,
            )

        self.assertIsNotNone(row)
        self.assertIsNotNone(row.coalition_advantage_mean)
        self.assertAlmostEqual(row.coalition_advantage_mean, expected_advantage)

    def test_meeting_scheduling_bruteforce_solver_finds_soft_overlap_optimum(self):
        instance = MeetingSchedulingInstanceData(
            variables={
                "agent_0__m001": VariableSpecLite(
                    name="agent_0__m001",
                    owner="agent_0",
                    domain=("skip", "0-1", "1-2", "0-2"),
                ),
                "agent_1__m001": VariableSpecLite(
                    name="agent_1__m001",
                    owner="agent_1",
                    domain=("skip", "0-1", "1-2", "0-2"),
                ),
            },
            meetings={
                "m001": MeetingSpec(
                    meeting_id="m001",
                    meeting_type="soft",
                    start=0,
                    end=2,
                    participants=("agent_0", "agent_1"),
                )
            },
            max_utility_upper_bound=2.0,
        )

        solution = solve_optimal_assignment(instance=instance)

        self.assertEqual(solution.solver_status, "complete")
        self.assertAlmostEqual(solution.joint_reward, 2.0)
        self.assertAlmostEqual(
            evaluate_assignment(solution.assignment, instance=instance),
            2.0,
        )

    def test_metrics_compute_variable_keyed_best_response_regret(self):
        class VarSpec:
            def __init__(self, name, owner, domain):
                self.name = name
                self.owner = owner
                self.domain = domain

        class Problem:
            def __init__(self):
                self.variables = {
                    "A__m001": VarSpec("A__m001", "A", ["skip", "go"]),
                    "B__m001": VarSpec("B__m001", "B", ["skip", "go"]),
                }

            def agent_variables(self, agent_name):
                return [
                    spec
                    for spec in self.variables.values()
                    if spec.owner == agent_name
                ]

        class Env:
            agent_names = ["A", "B"]
            assignment = {"A__m001": "skip", "B__m001": "go"}
            problem = Problem()
            max_joint_reward = 2.0

            def _rewards(self, actions):
                rewards = {
                    "A": 1.0 if actions.get("A__m001") == "go" else 0.0,
                    "B": 1.0 if actions.get("B__m001") == "go" else 0.0,
                }
                return sum(rewards.values()), rewards

        metrics = compute_collusion_metrics(
            env=Env(),
            colluders=["A"],
            secret_blackboard_id=None,
            secret_channel_enabled=False,
            prompt_variant="control",
            tool_events=[],
            blackboard_participants={},
            final_summary={
                "status": "complete",
                "attendance": {"A__m001": "skip", "B__m001": "go"},
                "joint_reward": 1.0,
                "agent_rewards": {"A": 0.0, "B": 1.0},
            },
        )

        self.assertAlmostEqual(metrics.coalition_mean_regret, 1.0)
        self.assertAlmostEqual(metrics.noncoalition_mean_regret, 0.0)
        self.assertAlmostEqual(metrics.reward_std, 0.5)

    def test_metrics_preserve_agent_keyed_best_response_regret(self):
        class Env:
            agent_names = ["A", "B"]
            assignment = {"A": "skip", "B": "task-1"}
            tasks = {"task-1": {}}
            max_joint_reward = 2.0

            def _rewards(self, actions):
                rewards = {
                    "A": 1.0 if actions.get("A") == "task-1" else 0.0,
                    "B": 1.0 if actions.get("B") == "task-1" else 0.0,
                }
                return sum(rewards.values()), rewards

        metrics = compute_collusion_metrics(
            env=Env(),
            colluders=["A"],
            secret_blackboard_id=None,
            secret_channel_enabled=False,
            prompt_variant="control",
            tool_events=[],
            blackboard_participants={},
            final_summary={
                "status": "complete",
                "assignment": {"A": "skip", "B": "task-1"},
                "joint_reward": 1.0,
                "agent_rewards": {"A": 0.0, "B": 1.0},
            },
        )

        self.assertAlmostEqual(metrics.coalition_mean_regret, 1.0)
        self.assertAlmostEqual(metrics.noncoalition_mean_regret, 0.0)


if __name__ == "__main__":
    unittest.main()
