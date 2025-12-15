"""
Utility functions for the communication protocol framework.

This module provides shared utility functions used across different environments
and components of the multi-agent communication protocol system.
"""

import shutil
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Union, Any, Dict, List, Optional
import re

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from envs.abstract_environment import AbstractEnvironment

def load_config(config_file) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config validation fails
    """
    # Resolve config path
    config_path = Path(config_file)
    if not config_path.is_absolute():
        if config_path.exists():
            config_path = config_path.resolve()
        else:
            # Try relative to script's parent directory
            import __main__
            config_path = Path(__main__.__file__).parent / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        config["_config_path"] = str(config_path.resolve())
    except OSError:
        config["_config_path"] = str(config_path)

    # Validate config structure
    validate_config(config)

    # Normalize simulation tag metadata to support multiple tags only
    simulation = config.get("simulation", {})
    raw_tags = simulation.get("tags")
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    elif isinstance(raw_tags, list):
        tags = [str(tag) for tag in raw_tags if tag is not None and str(tag).strip()]
    else:
        tags = []

    legacy_tag = simulation.pop("tag", None)
    if legacy_tag and not tags:
        tags = [legacy_tag]

    simulation["tags"] = tags

    return config

def load_seeds(seeds_file: str = "seeds.txt") -> List[int]:
    """Load simulation seeds from text file for reproducibility."""
    seeds_path = Path(seeds_file)
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seeds file {seeds_path} not found")

    with open(seeds_path, 'r') as f:
        seeds = [int(line.strip()) for line in f if line.strip()]

    return seeds


def prepare_simulation_config(base_config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Prepare a simulation config with the specified seed.

    Creates a copy of the base config and injects the seed into the simulation section.
    This ensures each simulation run has the correct seed for reproducibility.

    Args:
        base_config: Base configuration dictionary
        seed: Random seed for this simulation

    Returns:
        New config dictionary with seed injected into simulation.seed
    """
    import copy
    # Deep copy to avoid modifying the original config (important for parallel runs)
    sim_config = copy.deepcopy(base_config)
    sim_config["simulation"]["seed"] = seed
    return sim_config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and required fields.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If config structure is invalid or missing required sections
    """
    # Validate required top-level sections
    required_sections = ["simulation", "environment", "communication_network", "llm"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Config missing required sections: {missing_sections}")

    # Validate simulation section
    required_sim_keys = ["max_iterations", "max_planning_rounds", "seed"]
    missing_sim_keys = [key for key in required_sim_keys if key not in config["simulation"]]
    if missing_sim_keys:
        raise ValueError(f"Simulation section missing required keys: {missing_sim_keys}")

    # Validate communication_network section
    required_comm_keys = ["topology", "num_agents"]
    missing_comm_keys = [key for key in required_comm_keys if key not in config["communication_network"]]
    if missing_comm_keys:
        raise ValueError(f"Communication network section missing required keys: {missing_comm_keys}")

    # Validate LLM section
    if "provider" not in config["llm"]:
        raise ValueError("LLM section missing required key: provider")

    # Validate environment section has name
    if "name" not in config["environment"]:
        raise ValueError("Environment section missing required key: name")

    # Validate environment name points to an implemented environment class.
    env_name = config["environment"]["name"]
    implemented = set(get_implemented_environment_names())
    if env_name not in implemented:
        raise ValueError(
            f"Unknown environment.name={env_name!r}. "
            f"Expected one of: {sorted(implemented)!r}."
        )

    # Validate environments have exactly 1 iteration
    max_iterations = config["simulation"]["max_iterations"]
    if max_iterations != 1:
        raise ValueError(
            f"Environment '{env_name}' requires "
            f"max_iterations=1 in simulation config, but got {max_iterations}"
        )


def _sanitize_identifier(value: str) -> str:
    cleaned = value.lower().replace("/", "_").replace("-", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "model"


def _normalize_vllm_block(raw_vllm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize legacy vLLM config dictionaries into the new multi-model structure.
    """
    raw_vllm = raw_vllm or {}
    if isinstance(raw_vllm.get("models"), list):
        return raw_vllm

    model_id = (
        raw_vllm.get("model_id")
        or raw_vllm.get("id")
        or raw_vllm.get("served_model_name")
        or raw_vllm.get("model")
        or raw_vllm.get("model_name")
        or "model"
    )
    model_id = _sanitize_identifier(str(model_id))
    model_entry = {
        "id": model_id,
        "checkpoint": raw_vllm.get("model") or raw_vllm.get("model_name"),
        "served_model_name": raw_vllm.get("served_model_name"),
        "base_url": raw_vllm.get("base_url"),
        "host": raw_vllm.get("host", "127.0.0.1"),
        "port": raw_vllm.get("port", 8000),
        "tensor_parallel_size": raw_vllm.get("tensor_parallel_size"),
        "trust_remote_code": raw_vllm.get("trust_remote_code", False),
        "max_model_len": raw_vllm.get("max_model_len"),
        "dtype": raw_vllm.get("dtype"),
        "download_dir": raw_vllm.get("download_dir"),
        "gpu_memory_utilization": raw_vllm.get("gpu_memory_utilization"),
        "additional_args": raw_vllm.get("additional_args", []),
        "env": raw_vllm.get("env", {}),
        "request_timeout": raw_vllm.get("connection_timeout", 60),
        "api_key": raw_vllm.get("api_key"),
        "launch_command": raw_vllm.get("launch_command"),
    }
    normalized = dict(raw_vllm)
    normalized["models"] = [model_entry]
    return normalized


def _ensure_flag(args: List[str], flag: str) -> None:
    if flag not in args:
        args.append(flag)


def _ensure_flag_with_value(args: List[str], flag: str, value: str) -> None:
    try:
        idx = args.index(flag)
    except ValueError:
        args.extend([flag, value])
        return

    # If the flag exists but no value follows, append one
    if idx == len(args) - 1:
        args.append(value)
        return

    # Otherwise, replace the existing value
    args[idx + 1] = value


def _apply_vllm_model_presets(vllm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Inject model-specific defaults (tool parser, templates, etc.) when absent."""

    model_presets = [
        {
            "match": lambda name: "qwen" in name,
            "tool_call_parser": "hermes",
            "ensure_auto_tool_choice": True,
            "chat_template": None,
        },
    ]

    models = vllm_config.get("models") or []
    for model in models:
        name = (
            model.get("served_model_name")
            or model.get("checkpoint")
            or model.get("model")
            or model.get("id")
            or ""
        ).lower()

        args = list(model.get("additional_args", []))
        for preset in model_presets:
            if not preset["match"](name):
                continue
            if preset.get("ensure_auto_tool_choice"):
                _ensure_flag(args, "--enable-auto-tool-choice")
            parser = preset.get("tool_call_parser")
            if parser:
                _ensure_flag_with_value(args, "--tool-call-parser", parser)
            template = preset.get("chat_template")
            if template:
                _ensure_flag_with_value(args, "--chat-template", template)
            break

        model["additional_args"] = args

    return vllm_config


def _get_vllm_model_name(llm_config: Dict[str, Any]) -> str:
    vllm_block = _normalize_vllm_block(llm_config.get("vllm"))
    models = vllm_block.get("models") or []
    if not models:
        return "unknown-vllm-model"
    model = models[0]
    return (
        model.get("served_model_name")
        or model.get("checkpoint")
        or model.get("model")
        or model.get("id")
        or "unknown-vllm-model"
    )


def build_vllm_runtime(llm_config: Dict[str, Any]):
    """Construct a VLLMProviderRuntime from normalized configuration."""
    from llm_server.vllm.runtime import VLLMProviderRuntime

    normalized = _normalize_vllm_block(llm_config.get("vllm"))
    normalized = _apply_vllm_model_presets(normalized)
    return VLLMProviderRuntime(normalized)


def extract_model_info(full_config: Dict[str, Any]) -> str:
    """
    Extract model name from full config for logging.

    Args:
        full_config: Full configuration dictionary containing all simulation settings

    Returns:
        Model name string, or "unknown" if not found
    """
    if not full_config:
        return "unknown"

    llm_config = full_config.get("llm", {})
    provider = llm_config.get("provider", "unknown").lower()

    # Handle each provider using the new config format
    if provider == "openai":
        model_name = llm_config.get("openai", {}).get("model", "unknown")
    elif provider == "anthropic":
        model_name = llm_config.get("anthropic", {}).get("model", "unknown")
    elif provider == "gemini":
        model_name = llm_config.get("gemini", {}).get("model", "unknown")
    elif provider == "together":
        model_name = llm_config.get("together", {}).get("model", "unknown")
    elif provider == "vllm":
        model_name = _get_vllm_model_name(llm_config)
    else:
        model_name = "unknown"

    return model_name


def get_tag_model_subdir(full_config: Dict[str, Any]) -> str:
    """
    Generate tag_model subdirectory name from configuration.

    Args:
        full_config: Full configuration dictionary containing all simulation settings

    Returns:
        Formatted string: {tag}_{model_name}
    """
    if not full_config:
        return "unknown_unknown"

    # Get tag from simulation config
    simulation = full_config.get("simulation", {})
    tags = simulation.get("tags")
    primary_tag = None
    if isinstance(tags, list) and tags:
        primary_tag = tags[0]
    elif isinstance(tags, str):
        primary_tag = tags
    if not primary_tag:
        primary_tag = "unknown"

    # Get model name using existing function
    model_name = extract_model_info(full_config)

    return f"{primary_tag}_{model_name}"


def get_run_timestamp(full_config: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
    """Return the run timestamp stored in the simulation config, if any."""

    return full_config.get("simulation", {}).get("run_timestamp", default)


def _build_run_directory(root: str, environment_name: str, tag_model: str,
                         seed: Union[int, str], run_timestamp: Optional[str]) -> Path:
    parts = [Path(root), environment_name]
    if tag_model:
        parts.append(tag_model)
    if run_timestamp:
        parts.append(run_timestamp)
    parts.append(f"seed_{seed}")
    return Path(*parts)


def build_log_dir(environment_name: str, tag_model: str,
                  seed: Union[int, str], run_timestamp: Optional[str] = None) -> Path:
    """Build the filesystem path for log artifacts."""

    return _build_run_directory("logs", environment_name, tag_model, seed, run_timestamp)

def clear_seed_directories(environment_name: str, seed: Union[int, str], full_config: Dict[str, Any]) -> None:
    """
    Clear existing seed directories for both logs and plots to ensure clean state.

    Args:
        environment_name: Name of the environment (e.g., "Trading", "PersonalAssistant")
        seed: Seed value for the current run
        full_config: Full configuration dictionary containing all simulation settings
    """
    # Get tag_model subdirectory
    tag_model = get_tag_model_subdir(full_config)

    # Clear logs directory for this seed
    logs_seed_dir = Path(f"logs/{environment_name}/{tag_model}/seed_{seed}")
    if logs_seed_dir.exists():
        shutil.rmtree(logs_seed_dir)
        logger.warning(f"Cleared logs directory: {logs_seed_dir}")


def get_client_instance(llm_config: Dict[str, Any], *, agent_name: Optional[str] = None, vllm_runtime: Any = None):
    """
    Create and return the appropriate LLM client based on provider configuration.

    Args:
        llm_config: LLM configuration dictionary containing provider and provider-specific settings

    Returns:
        Instantiated client (OpenAIClient, AnthropicClient, GeminiClient, TogetherClient, or VLLMClient)

    Raises:
        ValueError: If provider is unknown
        NotImplementedError: If vllm provider is selected (currently not implemented)
    """
    provider = llm_config.get("provider", "vllm").lower()

    if provider == "openai":
        from llm_server.clients.openai_client import OpenAIClient
        return OpenAIClient()
    elif provider == "anthropic":
        from llm_server.clients.anthropic_client import AnthropicClient
        return AnthropicClient()
    elif provider == "gemini":
        from llm_server.clients.gemini_client import GeminiClient
        return GeminiClient()
    elif provider == "together":
        from llm_server.clients.together_client import TogetherClient

        together_cfg = llm_config.get("together") or {}
        base_url = str(together_cfg.get("base_url") or "https://api.together.xyz/v1")
        request_timeout = int(together_cfg.get("request_timeout", 60))
        api_key = together_cfg.get("api_key")
        return TogetherClient(base_url=base_url, api_key=api_key, request_timeout=request_timeout)
    elif provider == "vllm":
        if not vllm_runtime:
            raise ValueError("vLLM runtime is required to initialize vLLM clients")
        if agent_name is None:
            raise ValueError("agent_name must be provided when using vLLM provider")
        client, _ = vllm_runtime.create_client(agent_name)
        return client
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: openai, anthropic, gemini, together, vllm")


def _get_environment_registry():
    """Return the mapping of implemented environment class names -> classes."""
    # Import here to avoid circular dependencies / heavy imports at module import time.
    from envs.dcops.personal_assistant import PersonalAssistantEnvironment
    from envs.dcops.smart_grid import SmartGridEnvironment
    from envs.dcops.meeting_scheduling import MeetingSchedulingEnvironment

    return {
        MeetingSchedulingEnvironment.__name__: MeetingSchedulingEnvironment,
        PersonalAssistantEnvironment.__name__: PersonalAssistantEnvironment,
        SmartGridEnvironment.__name__: SmartGridEnvironment,
    }


def get_implemented_environment_names() -> List[str]:
    """Return the list of supported `environment.name` values."""
    return sorted(_get_environment_registry().keys())


def create_environment(protocol, environment_name: str, config, tool_logger) -> "AbstractEnvironment":
    """
    Create environment instance based on name.

    Args:
        protocol: Communication protocol instance
        environment_name: Environment class name (matches config `environment.name`)
        config: Configuration dictionary
        tool_logger: Tool call logger instance

    Returns:
        Instantiated environment object

    Raises:
        ValueError: If environment name is unknown
    """
    environments = _get_environment_registry()
    env_cls = environments.get(environment_name)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment.name={environment_name!r}. "
            f"Expected one of: {sorted(environments.keys())!r}."
        )
    return env_cls(protocol, config, tool_logger)


def get_model_name(provider: str, llm_config: Dict[str, Any]) -> str:
    """
    Extract model name based on provider from LLM configuration.

    Args:
        provider: LLM provider name (openai, anthropic, gemini, together, vllm)
        llm_config: LLM configuration dictionary

    Returns:
        Model name string

    Raises:
        ValueError: If provider is unknown
        NotImplementedError: If vllm provider is selected (currently not implemented)
    """
    # Extract model name based on provider
    if provider == "openai":
        model_name = llm_config.get("openai", {}).get("model", "gpt-4o")
    elif provider == "anthropic":
        model_name = llm_config.get("anthropic", {}).get("model", "claude-3-5-sonnet-20241022")
    elif provider == "gemini":
        model_name = llm_config.get("gemini", {}).get("model", "gemini-2.0-flash-exp")
    elif provider == "together":
        model_name = llm_config.get("together", {}).get("model", "unknown-together-model")
    elif provider == "vllm":
        model_name = _get_vllm_model_name(llm_config)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return model_name


def get_generation_params(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    provider = llm_config.get("provider", "").lower()
    provider_block = llm_config.get(provider, {}) if provider else {}
    if not isinstance(provider_block, dict):
        return {}

    raw_params = provider_block.get("params") or provider_block.get("generation") or {}
    if not isinstance(raw_params, dict):
        return {}

    # Never leak model identifiers into request params
    return {k: v for k, v in raw_params.items() if k != "model"}


def handle_mcp_connection_error(exc: Exception, url: str = "http://localhost:8000/mcp") -> bool:
    message = str(exc)
    connection_error = "Client failed to connect" in message or "All connection attempts failed" in message
    if connection_error:
        logger.error(
            f"Simulation aborted: could not connect to the MCP server at {url}. Start it in another terminal with `python src/server.py` and retry."
        )
        return True
    return False


def configure_logging(level: Optional[int] = None) -> None:
    import logging
    import os
    import sys

    if level is None:
        level = getattr(logging, os.getenv("TERRARIUM_LOG_LEVEL", "INFO").upper(), logging.INFO)

    if sys.stderr.isatty():
        colors = {
            logging.DEBUG: "\033[36m",
            logging.INFO: "\033[32m",
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[1;31m",
        }
        reset = "\033[0m"

        class _ColorFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
                record.log_color = colors.get(record.levelno, "")
                record.reset = reset
                return super().format(record)

        handler = logging.StreamHandler()
        handler.setFormatter(
            _ColorFormatter("%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s")
        )
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        logging.basicConfig(level=level, format="[%(levelname)s]: %(message)s", force=True)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
