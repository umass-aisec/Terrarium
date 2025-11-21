from __future__ import annotations

import atexit
import json
import os
import re
import shlex
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class ToolCallingRule:
    """Known vLLM tool/Reasoning presets keyed by substrings in the model id."""
    name: str  # Label so logs can say “using {name} parser”
    substrings: Tuple[str, ...]  # Lowercase tokens that match model ids/checkpoints
    parser: Optional[str] = None  
    chat_template: Optional[str] = None  # Required chat template path/name vLLM needs for tool calling instructions per model family
    reasoning_parser: Optional[str] = None
    enable_reasoning: bool = False
    supports_tool_calling: bool = True  # Some reasoning models don't support auto tool choice, we throw an error it this is False


# Derived from vLLM's tool-calling & reasoning-output docs:
# https://docs.vllm.ai/en/latest/features/tool_calling/
# https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
TOOL_CALLING_RULES: Tuple[ToolCallingRule, ...] = (
    ToolCallingRule(
        name="hermes",
        substrings=("hermes",),
        parser="hermes",
    ),
    ToolCallingRule(
        name="mistral",
        substrings=("mistral",),
        parser="mistral",
        chat_template="tool_chat_template_mistral_parallel.jinja",
    ),
    ToolCallingRule(
        name="granite-4.0",
        substrings=("granite-4.0", "granite-4"),
        parser="hermes",
    ),
    ToolCallingRule(
        name="granite-7b",
        substrings=("granite-7b",),
        parser="granite",
    ),
    ToolCallingRule(
        name="granite-3.1",
        substrings=("granite-3.1",),
        parser="granite",
    ),
    ToolCallingRule(
        name="granite-3.0",
        substrings=("granite-3.0",),
        parser="granite",
        chat_template="examples/tool_chat_template_granite.jinja",
    ),
    ToolCallingRule(
        name="granite-20b",
        substrings=("granite-20b",),
        parser="granite-20b-fc",
        chat_template="examples/tool_chat_template_granite_20b_fc.jinja",
    ),
    ToolCallingRule(
        name="internlm",
        substrings=("internlm", "internlm2.5", "internlm2_5"),
        parser="internlm",
        chat_template="examples/tool_chat_template_internlm2_tool.jinja",
    ),
    ToolCallingRule(
        name="jamba",
        substrings=("jamba", "ai21"),
        parser="jamba",
    ),
    ToolCallingRule(
        name="xlam-llama",
        substrings=("llama-xlam", "xlam-2-8b", "xlam-2-70b"),
        parser="xlam",
        chat_template="examples/tool_chat_template_xlam_llama.jinja",
    ),
    ToolCallingRule(
        name="xlam-qwen",
        substrings=("qwen-xlam", "xlam-1b", "xlam-3b", "xlam-32b"),
        parser="xlam",
        chat_template="examples/tool_chat_template_xlam_qwen.jinja",
    ),
    ToolCallingRule(
        name="qwen2.5",
        substrings=("qwen2.5", "qwen-2.5", "qwen2_5"),
        parser="hermes",
    ),
    ToolCallingRule(
        name="qwen3",
        substrings=("qwen3", "qwen 3"),
        parser="hermes",
        reasoning_parser="qwen3",
    ),
    ToolCallingRule(
        name="qwq-32b",
        substrings=("qwq", "qwq-32b"),
        parser="hermes",
        reasoning_parser="deepseek_r1",
    ),
    ToolCallingRule(
        name="minimax-m1",
        substrings=("minimax", "minimaxai"),
        parser="minimax",
        chat_template="examples/tool_chat_template_minimax_m1.jinja",
    ),
    ToolCallingRule(
        name="deepseek-v3",
        substrings=("deepseek-v3", "deepseek-v3-0324", "deepseek-r1-0528"),
        parser="deepseek_v3",
        chat_template="examples/tool_chat_template_deepseekv3.jinja",
    ),
    ToolCallingRule(
        name="deepseek-v31",
        substrings=("deepseek-v3.1", "deepseek v3.1"),
        parser="deepseek_v31",
        chat_template="examples/tool_chat_template_deepseekv31.jinja",
    ),
    ToolCallingRule(
        name="deepseek-r1",
        substrings=("deepseek r1", "deepseek-r1"),
        reasoning_parser="deepseek_r1",
        supports_tool_calling=False,
    ),
    ToolCallingRule(
        name="kimi-k2",
        substrings=("kimi-k2", "moonshotai/kimi", "moonshot kimi"),
        parser="kimi_k2",
    ),
    ToolCallingRule(
        name="hunyuan-a13b",
        substrings=("hunyuan-a13b", "hunyuan a13b"),
        parser="hunyuan_a13b",
        reasoning_parser="hunyuan_a13b",
    ),
    ToolCallingRule(
        name="glm-4.5",
        substrings=("glm-4.5", "glm45", "glm_4.5"),
        parser="glm45",
        reasoning_parser="glm45",
    ),
    ToolCallingRule(
        name="glm-4-9b",
        substrings=("glm-4-9b", "glm4-9b"),
        parser="glm45",
        reasoning_parser="glm45",
    ),
    ToolCallingRule(
        name="ernie-4.5",
        substrings=("ernie-4.5", "ernie45"),
        reasoning_parser="ernie45",
        supports_tool_calling=False,
    ),
)


def _match_tool_calling_rule(model_label: str) -> Optional[ToolCallingRule]:
    """Return the first tool-calling rule whose substring matches the label."""
    normalized = (model_label or "").lower()
    for rule in TOOL_CALLING_RULES:
        if any(token in normalized for token in rule.substrings):
            return rule
    return None


def _resolve_chat_template_candidate(template: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a chat-template path distributed with vLLM if needed."""
    if not template:
        return None, None

    candidate = template.strip()
    if not candidate:
        return None, None
    if candidate in {"tool_use"}:
        return candidate, None

    path = Path(candidate).expanduser()
    search_paths: List[Path] = []

    if path.is_absolute():
        search_paths.append(path)
    else:
        search_paths.append(Path.cwd() / path)

    try:
        import vllm  # type: ignore
    except Exception:
        vllm_root: Optional[Path] = None
    else:
        vllm_root = Path(vllm.__file__).resolve().parent
        search_paths.append(vllm_root / candidate)

    for entry in search_paths:
        if entry.exists():
            return str(entry), None

    return None, f"Chat template '{candidate}' was not found relative to CWD or the vLLM package."


def _sanitize_model_id(value: str) -> str:
    cleaned = value.lower().replace("/", "_").replace("-", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "model"


def _derive_model_id(spec: Dict[str, Any]) -> str:
    candidate = (
        spec.get("checkpoint")
        or spec.get("served_model_name")
        or spec.get("model")
        or spec.get("model_name")
    )
    if not candidate:
        raise ValueError(
            "Each vLLM model spec must include a checkpoint (or served_model_name/model/model_name) "
            "to derive a model identifier."
        )
    return _sanitize_model_id(str(candidate))


def _normalize_base_url(base_url: Optional[str], host: str, port: int) -> str:
    """Return a normalized base URL ending with /v1."""
    if base_url:
        normalized = base_url.rstrip("/")
    else:
        normalized = f"http://{host}:{port}/v1"
    return normalized


@dataclass
class VLLMModelSpec:
    """Per-model serving configuration."""

    id: str
    checkpoint: Optional[str] = None
    served_model_name: Optional[str] = None
    base_url: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: Optional[int] = None
    trust_remote_code: bool = False
    max_model_len: Optional[int] = None
    dtype: Optional[str] = None
    download_dir: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    additional_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    request_timeout: int = 60
    api_key: Optional[str] = None
    launch_command: Optional[Any] = None
    auto_configure_tool_choice: Optional[bool] = None
    enable_auto_tool_choice: Optional[bool] = None
    tool_call_parser: Optional[str] = None
    chat_template: Optional[str] = None
    tool_parser_plugin: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_reasoning: Optional[bool] = None

    def api_base(self) -> str:
        return _normalize_base_url(self.base_url, self.host, self.port)

    def served_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        if self.checkpoint:
            return self.checkpoint
        return self.id


@dataclass
class VLLMGlobalConfig:
    """Global configuration shared across all vLLM model specs."""

    models: Dict[str, VLLMModelSpec]
    primary_model_id: str
    auto_start_server: bool = False
    auto_configure_tool_choice: bool = True
    startup_timeout: int = 180
    health_check_path: str = "/v1/models"
    health_poll_interval: float = 2.0
    api_key: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw_config: Dict[str, Any]) -> "VLLMGlobalConfig":
        if not raw_config:
            raise ValueError("Missing llm.vllm configuration block")

        models_list = raw_config.get("models")
        if not models_list:
            raise ValueError("llm.vllm.models must contain at least one model spec")

        model_map: Dict[str, VLLMModelSpec] = {}
        for spec in models_list:
            model_id = _derive_model_id(spec)
            model = VLLMModelSpec(
                id=model_id,
                checkpoint=spec.get("checkpoint") or spec.get("model") or spec.get("model_name"),
                served_model_name=spec.get("served_model_name") or spec.get("served_name"),
                base_url=spec.get("base_url"),
                host=spec.get("host", "127.0.0.1"),
                port=int(spec.get("port", 8000)),
                tensor_parallel_size=spec.get("tensor_parallel_size"),
                trust_remote_code=bool(spec.get("trust_remote_code", False)),
                max_model_len=spec.get("max_model_len"),
                dtype=spec.get("dtype"),
                download_dir=spec.get("download_dir"),
                gpu_memory_utilization=spec.get("gpu_memory_utilization"),
                additional_args=list(spec.get("additional_args", [])),
                env=dict(spec.get("env") or {}),
                request_timeout=int(spec.get("request_timeout", spec.get("connection_timeout", 60))),
                api_key=spec.get("api_key"),
                launch_command=spec.get("launch_command"),
                auto_configure_tool_choice=spec.get("auto_configure_tool_choice"),
                enable_auto_tool_choice=spec.get("enable_auto_tool_choice"),
                tool_call_parser=spec.get("tool_call_parser"),
                chat_template=spec.get("chat_template"),
                tool_parser_plugin=spec.get("tool_parser_plugin"),
                reasoning_parser=spec.get("reasoning_parser"),
                enable_reasoning=spec.get("enable_reasoning"),
            )
            model_map[model.id] = model

        if len(model_map) == 0:
            raise ValueError("llm.vllm.models must contain at least one model spec")
        if len(model_map) > 1:
            raise ValueError(
                "llm.vllm.models now supports exactly one model spec because per-agent routing was removed."
            )
        primary_model_id = next(iter(model_map.keys()))
        legacy_default = raw_config.get("default_model")
        if legacy_default and legacy_default != primary_model_id:
            raise ValueError(
                "Multiple vLLM models are no longer supported. Remove the legacy default_model setting "
                "and keep a single entry under llm.vllm.models."
            )
        if legacy_default:
            warnings.warn(
                "llm.vllm.default_model is deprecated; remove it to simplify the config.",
                stacklevel=2,
            )

        return cls(
            models=model_map,
            primary_model_id=primary_model_id,
            auto_start_server=bool(raw_config.get("auto_start_server", False)),
            auto_configure_tool_choice=bool(raw_config.get("auto_configure_tool_choice", True)),
            startup_timeout=int(raw_config.get("startup_timeout", raw_config.get("connection_timeout", 180))),
            health_check_path=str(raw_config.get("health_check_path", "/v1/models")),
            health_poll_interval=float(raw_config.get("health_poll_interval", 2.0)),
            api_key=raw_config.get("api_key"),
            env=dict(raw_config.get("env") or {}),
        )

    def get_model_for_agent(self, agent_name: str) -> VLLMModelSpec:
        del agent_name  # unused, but preserved for API compatibility
        if self.primary_model_id not in self.models:
            raise ValueError(
                f"Configured vLLM model id '{self.primary_model_id}' is missing from llm.vllm.models"
            )
        return self.models[self.primary_model_id]


class VLLMServerManager:
    """Bootstraps and monitors vLLM server processes."""

    def __init__(self, config: VLLMGlobalConfig):
        self.config = config
        self._process_table: Dict[str, Tuple[subprocess.Popen, Optional[Any]]] = {}
        atexit.register(self.shutdown)

    def _stop_process(self, model_id: str, *, wait_seconds: float = 5.0) -> None:
        """Gracefully stop a managed vLLM process, waiting up to wait_seconds."""
        process_entry = self._process_table.get(model_id)
        if not process_entry:
            return
        process, log_file = process_entry
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
        if log_file:
            try:
                log_file.write(f"\n=== Stopping {model_id} ===\n")
                log_file.close()
            except Exception:
                pass
        self._process_table.pop(model_id, None)

    def _stop_all_servers(self, *, except_id: Optional[str] = None, wait_seconds: float = 5.0) -> None:
        """Stop every tracked server except the optional except_id."""
        for model_id in list(self._process_table.keys()):
            if except_id and model_id == except_id:
                continue
            self._stop_process(model_id, wait_seconds=wait_seconds)

    @staticmethod
    def _health_url(spec: VLLMModelSpec, config: VLLMGlobalConfig) -> str:
        base = spec.api_base().rstrip("/")
        path = config.health_check_path
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}"

    @staticmethod
    def _is_server_alive(url: str, timeout: float = 2.0) -> bool:
        try:
            resp = requests.get(url, timeout=timeout)
            return resp.ok
        except requests.RequestException:
            return False

    def _apply_tool_and_reasoning_settings(self, spec: VLLMModelSpec, cmd: List[str]) -> None:
        """Inject --tool-call-parser/--reasoning-parser flags when possible."""
        auto_cfg = spec.auto_configure_tool_choice
        if auto_cfg is None:
            auto_cfg = self.config.auto_configure_tool_choice

        enable_auto_tool_choice = spec.enable_auto_tool_choice
        if enable_auto_tool_choice is None:
            enable_auto_tool_choice = auto_cfg

        matched_rule: Optional[ToolCallingRule] = None
        label_candidates = [
            spec.served_model_name or "",
            spec.checkpoint or "",
            spec.id or "",
        ]
        if auto_cfg:
            for candidate in label_candidates:
                rule = _match_tool_calling_rule(candidate)
                if rule:
                    matched_rule = rule
                    break

        tool_call_parser = spec.tool_call_parser or (matched_rule.parser if matched_rule else None)
        reasoning_parser = spec.reasoning_parser or (matched_rule.reasoning_parser if matched_rule else None)
        chat_template = spec.chat_template or (matched_rule.chat_template if matched_rule else None)

        if enable_auto_tool_choice:
            if matched_rule and not matched_rule.supports_tool_calling and not spec.tool_call_parser:
                raise RuntimeError(
                    f"Model '{spec.id}' is not listed as tool-calling capable in vLLM's docs "
                    "and Terrarium requires structured tool calls. Pick a different checkpoint or "
                    "override llm.vllm.models[].tool_call_parser with a known-compatible parser "
                    "(see https://docs.vllm.ai/en/latest/features/tool_calling/)."
                )
            elif not tool_call_parser and not spec.tool_parser_plugin:
                warnings.warn(
                    f"Auto tool calling requested for '{spec.id}', but no tool_call_parser could be resolved. "
                    "Disabling --enable-auto-tool-choice for this launch. "
                    "Set llm.vllm.models[].tool_call_parser explicitly if the model supports tools."
                )
                enable_auto_tool_choice = False

        if enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")

        if tool_call_parser:
            cmd += ["--tool-call-parser", tool_call_parser]

        if spec.tool_parser_plugin:
            cmd += ["--tool-parser-plugin", spec.tool_parser_plugin]

        if chat_template:
            resolved_template, template_warning = _resolve_chat_template_candidate(chat_template)
            if resolved_template:
                cmd += ["--chat-template", resolved_template]
            elif template_warning:
                warnings.warn(
                    f"{template_warning} (model '{spec.id}'). "
                    "Continuing without an explicit chat template; vLLM will use model defaults."
                )

        if spec.enable_reasoning:
            cmd.append("--enable-reasoning")
        if reasoning_parser:
            cmd += ["--reasoning-parser", reasoning_parser]

        if auto_cfg and not matched_rule and not spec.tool_call_parser:
                warnings.warn(
                    f"Could not auto-configure tool calling for '{spec.id}' because it is not in Terrarium's "
                    "supported model list. Refer to https://docs.vllm.ai/en/latest/features/tool_calling/ "
                    "and set llm.vllm.models[].tool_call_parser manually."
                )

    def ensure_server(self, spec: VLLMModelSpec) -> None:
        """Ensure a server for the given model is reachable, launching if needed."""
        # Stop any previous servers so we can spin up the new model cleanly.
        self._stop_all_servers(except_id=spec.id, wait_seconds=5.0)

        health_url = self._health_url(spec, self.config)
        if self._is_server_alive(health_url):
            return

        if not self.config.auto_start_server:
            raise RuntimeError(
                f"vLLM server for model '{spec.id}' is not reachable at {health_url} "
                "and auto_start_server is disabled."
            )

        self._launch_process(spec)
        self._wait_until_ready(spec, health_url)

    def _launch_process(self, spec: VLLMModelSpec) -> None:
        if spec.id in self._process_table and self._process_table[spec.id][0].poll() is None:
            return  # already running

        if spec.launch_command:
            if isinstance(spec.launch_command, str):
                cmd = shlex.split(spec.launch_command)
            else:
                cmd = list(spec.launch_command)
        else:
            if not spec.checkpoint:
                raise ValueError(
                    f"Model '{spec.id}' requires 'checkpoint' (or explicit launch_command) to auto-start vLLM"
                )
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                spec.checkpoint,
                "--port",
                str(spec.port),
                "--host",
                spec.host,
            ]
            if spec.served_model_name:
                cmd += ["--served-model-name", spec.served_model_name]
            if spec.tensor_parallel_size:
                cmd += ["--tensor-parallel-size", str(spec.tensor_parallel_size)]
            if spec.max_model_len:
                cmd += ["--max-model-len", str(spec.max_model_len)]
            if spec.dtype:
                cmd += ["--dtype", spec.dtype]
            if spec.download_dir:
                cmd += ["--download-dir", spec.download_dir]
            if spec.gpu_memory_utilization:
                cmd += ["--gpu-memory-utilization", str(spec.gpu_memory_utilization)]
            if spec.trust_remote_code:
                cmd.append("--trust-remote-code")
            self._apply_tool_and_reasoning_settings(spec, cmd)
            if spec.additional_args:
                cmd += list(spec.additional_args)

        env = os.environ.copy()
        env.update(self.config.env or {})
        env.update(spec.env or {})

        log_dir = Path("logs") / "vllm"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{spec.id}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"=== Launching vLLM model {spec.id} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env,
        )
        self._process_table[spec.id] = (process, log_file)

    def _wait_until_ready(self, spec: VLLMModelSpec, health_url: str) -> None:
        """Poll the health endpoint until it responds or timeout is reached."""
        deadline = time.time() + self.config.startup_timeout
        process, _ = self._process_table.get(spec.id, (None, None))

        while time.time() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server for model '{spec.id}' exited prematurely with code {process.returncode}"
                )
            if self._is_server_alive(health_url):
                return
            time.sleep(self.config.health_poll_interval)

        raise TimeoutError(
            f"Timed out waiting for vLLM server '{spec.id}' to become ready at {health_url}"
        )

    def shutdown(self) -> None:
        """Terminate any managed vLLM processes."""
        self._stop_all_servers(wait_seconds=5.0)


class VLLMProviderRuntime:
    """High level helper that ties configuration, server manager, and clients together."""

    def __init__(self, raw_config: Dict[str, Any]):
        self.global_config = VLLMGlobalConfig.from_dict(raw_config)
        self.server_manager = VLLMServerManager(self.global_config)

    def describe_default_model(self) -> str:
        default_spec = self.global_config.models[self.global_config.primary_model_id]
        return default_spec.served_name()

    def describe_log_path(self) -> str:
        """Return the expected log file path for the active vLLM model."""
        spec = self.global_config.get_model_for_agent("unused")
        return str(Path("logs") / "vllm" / f"{spec.id}.log")

    def create_client(self, agent_name: str):
        """Return (client, served_model_name) for a specific agent."""
        from llm_server.clients.vllm_client import VLLMClient

        spec = self.global_config.get_model_for_agent(agent_name)
        self.server_manager.ensure_server(spec)

        client = VLLMClient(
            base_url=spec.api_base(),
            model_name=spec.served_name(),
            api_key=spec.api_key or self.global_config.api_key or "EMPTY",
            request_timeout=spec.request_timeout,
        )
        return client, spec.served_name()

    def shutdown(self) -> None:
        self.server_manager.shutdown()
