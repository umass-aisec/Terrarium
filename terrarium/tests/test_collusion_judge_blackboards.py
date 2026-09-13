import unittest
import json
import tempfile
from pathlib import Path

try:
    from experiments.collusion.judge_paths import derive_judge_output_tag, judge_dir_name
    from experiments.collusion.plots.generate_regret_report import (
        _build_plot_header,
        _judge_mean_rating,
    )
    from experiments.collusion.plots.plot_judge_vs_coalition_advantage import (
        _title_from_csv_metadata,
    )
    from experiments.collusion.plots.plot_sweep import build_rows
    from experiments.collusion.judge_blackboards import (
        _apply_judge_overrides,
        _format_blackboard_entry_transcript,
        _judge_model_banner_lines,
        resolve_judge_config,
    )
    from experiments.common.plotting.load_runs import LoadedRun
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("experiments"):
        raise unittest.SkipTest("Optional experiments package is not available") from exc
    raise


class CollusionJudgeConfigTests(unittest.TestCase):
    def test_format_blackboard_entry_transcript_reads_hospital_broadcast_actions(self):
        transcript, participants, has_communication = _format_blackboard_entry_transcript(
            {
                "blackboard_id": "1",
                "participants": ["AgentA", "AgentB"],
                "events": [
                    {
                        "agent": "SYSTEM",
                        "kind": "context",
                        "payload": {"message": "Private coordination board."},
                    },
                    {
                        "agent": "AgentA",
                        "kind": "action_executed",
                        "payload": {
                            "action_params": {
                                "broadcast_message": {
                                    "AgentA": {
                                        "message": "SURPLUS PPE at General_Hospital.",
                                        "rationale": "Unused inventory is available.",
                                    }
                                }
                            }
                        },
                    },
                ],
            }
        )

        self.assertEqual(participants, ["AgentA", "AgentB"])
        self.assertTrue(has_communication)
        self.assertIn("SYSTEM: Private coordination board.", transcript)
        self.assertIn("AgentA: SURPLUS PPE at General_Hospital.", transcript)

    def test_resolve_judge_config_reuses_matching_model_profile(self):
        experiment_cfg = {
            "llm_models": [
                {
                    "label": "gpt-5.4",
                    "llm": {
                        "provider": "foundry",
                        "foundry": {
                            "model": "gpt-5.4",
                            "base_model": "gpt-5.4",
                            "params": {
                                "max_tokens": 1500,
                                "temperature": 0.7,
                                "reasoning_effort": "minimal",
                                "verbosity": "low",
                                "tool_choice": "required",
                            },
                        },
                    },
                }
            ]
        }
        run_cfg = {
            "model_label": "gpt-5.4",
            "provider": "foundry",
            "model": "gpt-5.4",
        }

        judge_cfg = resolve_judge_config(
            run_cfg=run_cfg,
            experiment_cfg=experiment_cfg,
            model_llm_map={"gpt-5.4": experiment_cfg["llm_models"][0]["llm"]},
        )

        self.assertEqual(judge_cfg.provider, "foundry")
        self.assertEqual(judge_cfg.model, "gpt-5.4")
        self.assertEqual(judge_cfg.request_params["reasoning_effort"], "minimal")
        self.assertEqual(judge_cfg.request_params["verbosity"], "low")
        self.assertEqual(judge_cfg.request_params["max_output_tokens"], 256)
        self.assertEqual(judge_cfg.request_params["temperature"], 0.0)
        self.assertNotIn("tool_choice", judge_cfg.request_params)

    def test_resolve_judge_config_falls_back_to_run_summary(self):
        run_cfg = {
            "model_label": "fw-glm-5",
            "provider": "foundry",
            "model": "FW-GLM-5",
        }

        judge_cfg = resolve_judge_config(run_cfg=run_cfg)

        self.assertEqual(judge_cfg.provider, "foundry")
        self.assertEqual(judge_cfg.model, "FW-GLM-5")
        self.assertEqual(judge_cfg.request_params["max_output_tokens"], 256)
        self.assertEqual(judge_cfg.request_params["temperature"], 0.0)

    def test_resolve_judge_config_supports_provider_and_model_overrides(self):
        run_cfg = {
            "model_label": "openai-gpt-4o-mini",
            "provider": "openai",
            "model": "gpt-4o-mini",
        }

        judge_cfg = resolve_judge_config(
            run_cfg=run_cfg,
            judge_provider="foundry",
            judge_model="gpt-4.1-mini-2025-04-14",
            max_output_tokens=128,
            temperature=0.2,
        )

        self.assertEqual(judge_cfg.provider, "foundry")
        self.assertEqual(judge_cfg.model, "gpt-4.1-mini-2025-04-14")
        self.assertEqual(judge_cfg.request_params["max_output_tokens"], 128)
        self.assertEqual(judge_cfg.request_params["temperature"], 0.2)

    def test_resolve_judge_config_supports_foundry_env_var_overrides(self):
        experiment_cfg = {
            "llm_models": [
                {
                    "label": "claude-opus-4-6",
                    "llm": {
                        "provider": "foundry",
                        "foundry": {
                            "project_endpoint_env_var": "AI_FOUNDRY_RBR_EAST_US_2_PROJECT_ENDPOINT",
                            "api_key_env_var": "AI_FOUNDRY_RBR_EAST_US_2_API_KEY",
                            "model": "claude-opus-4-6",
                        },
                    },
                }
            ]
        }
        run_cfg = {
            "model_label": "claude-opus-4-6",
            "provider": "foundry",
            "model": "claude-opus-4-6",
        }

        judge_cfg = resolve_judge_config(
            run_cfg=run_cfg,
            experiment_cfg=experiment_cfg,
            model_llm_map={"claude-opus-4-6": experiment_cfg["llm_models"][0]["llm"]},
            judge_provider="foundry",
            judge_model="gpt-5.4",
            judge_project_endpoint_env_var="TEMP_FOUNDRY_PROJECT_ENDPOINT",
            judge_api_key_env_var="TEMP_FOUNDRY_API_KEY",
            judge_auth_mode="api_key",
        )

        self.assertEqual(judge_cfg.provider, "foundry")
        self.assertEqual(judge_cfg.model, "gpt-5.4")

    def test_judge_model_override_drops_inherited_foundry_api_style(self):
        llm_cfg = _apply_judge_overrides(
            base_llm_cfg={
                "provider": "foundry",
                "foundry": {
                    "project_endpoint_env_var": "AI_FOUNDRY_PROJECT_ENDPOINT",
                    "api_key_env_var": "AI_FOUNDRY_API_KEY",
                    "api_style": "chat_completions",
                    "base_model": "grok-4-20-reasoning",
                    "model": "grok-4-20-reasoning",
                    "params": {"max_tokens": 1500},
                },
            },
            run_cfg={
                "model_label": "grok-4-20-reasoning",
                "provider": "foundry",
                "model": "grok-4-20-reasoning",
            },
            judge_provider="foundry",
            judge_model="claude-opus-4-6",
        )

        self.assertEqual(llm_cfg["foundry"]["model"], "claude-opus-4-6")
        self.assertNotIn("api_style", llm_cfg["foundry"])
        self.assertNotIn("base_model", llm_cfg["foundry"])
        self.assertEqual(llm_cfg["foundry"]["params"], {"max_tokens": 1500})

    def test_judge_model_banner_lines_report_resolved_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "gpt-5.4" / "complete_n9_c4" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_label": "gpt-5.4",
                        "provider": "foundry",
                        "model": "gpt-5.4",
                    }
                ),
                encoding="utf-8",
            )

            lines = _judge_model_banner_lines(run_dirs=[run_dir])

        self.assertEqual(lines, ["Judge model: gpt-5.4"])

    def test_banner_and_helper_support_tagged_judge_outputs(self):
        tag = derive_judge_output_tag(
            judge_provider="foundry",
            judge_model="gpt-4.1-mini-2025-04-14",
        )
        self.assertEqual(
            tag,
            "foundry__gpt-4.1-mini-2025-04-14",
        )
        self.assertEqual(
            judge_dir_name(tag),
            "judge_secret_blackboard__foundry__gpt-4.1-mini-2025-04-14",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "gpt-5.4" / "complete_n9_c4" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_label": "gpt-5.4",
                        "provider": "foundry",
                        "model": "gpt-5.4",
                    }
                ),
                encoding="utf-8",
            )

            lines = _judge_model_banner_lines(run_dirs=[run_dir], judge_output_tag=tag)

        self.assertEqual(
            lines,
            [
                "Judge output dir: judge_secret_blackboard__foundry__gpt-4.1-mini-2025-04-14",
                "Judge model: gpt-5.4",
            ],
        )

    def test_plot_readers_can_load_tagged_judge_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "gpt-5.4" / "complete_n9_c4" / "run-a"
            run_dir.mkdir(parents=True)
            judge_dir = (
                run_dir.parent.parent
                / judge_dir_name("foundry__gpt41mini")
                / run_dir.parent.name
            )
            judge_dir.mkdir(parents=True)
            (judge_dir / f"{run_dir.name}.json").write_text(
                json.dumps(
                    {
                        "judgements": {
                            "simple": {"rating": 1},
                            "medium": {"rating": 3},
                            "complex": {"rating": 5},
                        }
                    }
                ),
                encoding="utf-8",
            )

            loaded_run = LoadedRun(
                run_dir=run_dir,
                run_config={
                    "run_id": "run-a",
                    "seed": 1,
                    "replica_index": 0,
                    "topology": "complete",
                    "colluder_count": 4,
                    "secret_channel_enabled": True,
                    "prompt_variant": "simple",
                },
                final_summary={"joint_reward": 10.0, "joint_reward_ratio": 0.5},
                metrics={"coalition_mean_regret": 0.1, "noncoalition_mean_regret": 0.3},
                judge_results=None,
                survey_responses=None,
                tool_events=None,
                agent_turns=None,
                blackboards=None,
            )

            rows = build_rows([loaded_run], judge_output_tag="foundry__gpt41mini")
            rating = _judge_mean_rating(
                model_dir=run_dir.parent.parent,
                sweep_name=run_dir.parent.name,
                run_name=run_dir.name,
                judge_output_tag="foundry__gpt41mini",
            )

        self.assertEqual(rows[0]["judge_mean_rating"], 3.0)
        self.assertEqual(rating, 3.0)

    def test_regret_report_header_includes_environment_and_judge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.json").write_text(
                json.dumps({"environment": {"name": "JiraTicketEnvironment"}}),
                encoding="utf-8",
            )
            judge_dir = (
                root
                / "runs"
                / "model-a"
                / judge_dir_name("foundry__gpt-5.4-nano")
                / "complete_n6_c2"
            )
            judge_dir.mkdir(parents=True)
            (judge_dir / "run-a.json").write_text(
                json.dumps(
                    {
                        "judge_config": {
                            "provider": "foundry",
                            "model": "gpt-5.4-nano",
                        }
                    }
                ),
                encoding="utf-8",
            )

            header = _build_plot_header(
                root=root,
                sweep_name="complete_n6_c2",
                judge_output_tag="foundry__gpt-5.4-nano",
            )

        self.assertEqual(
            header,
            "Environment: JiraTicketEnvironment | Judge: foundry / gpt-5.4-nano",
        )

    def test_scatter_title_uses_regret_report_csv_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "regret_report__normalized_regret__coalition_gap__judge__data.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "model_label,condition,metric_key,mean,plot_header",
                        "gpt-5.4,control,judge_mean_rating,3.0,Environment: JiraTicketEnvironment | Judge: foundry / gpt-5.4-nano",
                    ]
                ),
                encoding="utf-8",
            )

            title = _title_from_csv_metadata(csv_path)

        self.assertEqual(
            title,
            "Environment: JiraTicketEnvironment | Judge: foundry / gpt-5.4-nano",
        )


if __name__ == "__main__":
    unittest.main()
