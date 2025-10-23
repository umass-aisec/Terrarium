"""
Utility functions for the communication protocol framework.

This module provides shared utility functions used across different environments
and components of the multi-agent communication protocol system.
"""

import shutil
import yaml
from pathlib import Path
from typing import Union, Any, Dict, List


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

    # Validate config structure
    validate_config(config)

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

    Creates a copy of the base config and injects the seed into the environment section.
    This ensures each simulation run has the correct seed for reproducibility.

    Args:
        base_config: Base configuration dictionary
        seed: Random seed for this simulation

    Returns:
        New config dictionary with seed injected into environment.rng_seed
    """
    import copy
    # Deep copy to avoid modifying the original config (important for parallel runs)
    sim_config = copy.deepcopy(base_config)
    sim_config["environment"]["rng_seed"] = seed
    sim_config["_current_seed"] = seed  # Track current seed at top level for convenience
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
    required_sections = ["simulation", "environment", "llm"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Config missing required sections: {missing_sections}")

    # Validate simulation section
    required_sim_keys = ["max_iterations", "max_planning_rounds"]
    missing_sim_keys = [key for key in required_sim_keys if key not in config["simulation"]]
    if missing_sim_keys:
        raise ValueError(f"Simulation section missing required keys: {missing_sim_keys}")

    # Validate LLM section
    if "provider" not in config["llm"]:
        raise ValueError("LLM section missing required key: provider")

    # Validate environment section has name
    if "name" not in config["environment"]:
        raise ValueError("Environment section missing required key: name")

    # Validate DCOP environments have exactly 1 iteration
    dcop_environments = ["MeetingScheduling", "PersonalAssistant", "SmartGrid"]
    if config["environment"]["name"] in dcop_environments:
        max_iterations = config["simulation"].get("max_iterations", 3)
        if max_iterations != 1:
            raise ValueError(
                f"DCOP environment '{config['environment']['name']}' requires "
                f"max_iterations=1 in simulation config, but got {max_iterations}"
            )


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
    elif provider == "vllm":
        raise NotImplementedError
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
    tag = full_config.get("simulation", {}).get("tag", "unknown")

    # Get model name using existing function
    model_name = extract_model_info(full_config)

    return f"{tag}_{model_name}"


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

    # Clear plots directory for this seed
    plots_seed_dir = Path(f"plots/{environment_name}/{tag_model}/seed_{seed}")
    if plots_seed_dir.exists():
        shutil.rmtree(plots_seed_dir)
        print(f"Cleared plots directory: {plots_seed_dir}")

    # Clear logs directory for this seed
    logs_seed_dir = Path(f"logs/{environment_name}/{tag_model}/seed_{seed}")
    if logs_seed_dir.exists():
        shutil.rmtree(logs_seed_dir)
        print(f"Cleared logs directory: {logs_seed_dir}")


def get_client_instance(llm_config: Dict[str, Any]):
    """
    Create and return the appropriate LLM client based on provider configuration.

    Args:
        llm_config: LLM configuration dictionary containing provider and provider-specific settings

    Returns:
        Instantiated client (OpenAIClient, AnthropicClient, GeminiClient, or VLLMClient)

    Raises:
        ValueError: If provider is unknown
        NotImplementedError: If vllm provider is selected (currently not implemented)
    """
    # Import here to avoid circular dependencies
    from llm_server.clients.openai import OpenAIClient
    from llm_server.clients.anthropic_client import AnthropicClient
    from llm_server.clients.gemini_client import GeminiClient

    provider = llm_config.get("provider", "vllm").lower()

    if provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    elif provider == "gemini":
        return GeminiClient()
    elif provider == "vllm":
        raise NotImplementedError("vLLM provider is currently not implemented")
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: openai, anthropic, gemini, vllm")


def create_environment(protocol, environment_name: str, config, tool_logger):
    """
    Create environment instance based on name.

    Args:
        protocol: Communication protocol instance
        environment_name: Name of the environment to create
        config: Configuration dictionary
        tool_logger: Tool call logger instance

    Returns:
        Instantiated environment object

    Raises:
        ValueError: If environment name is unknown
    """
    # Import here to avoid circular dependencies
    from envs.negotiation.trading import TradingGameEnvironment
    from envs.dcops.personal_assistant import PersonalAssistantEnvironment
    from envs.dcops.smart_grid import SmartGridEnvironment
    from envs.dcops.meeting_scheduling import MeetingSchedulingEnvironment

    environments = {
        "PersonalAssistant": PersonalAssistantEnvironment,
        "Trading": TradingGameEnvironment,
        "SmartGrid": SmartGridEnvironment,
        "MeetingScheduling": MeetingSchedulingEnvironment,
    }
    if environment_name not in environments:
        raise ValueError(f"Unknown environment: {environment_name}")
    return environments[environment_name](protocol, config, tool_logger)


def get_model_name(provider: str, llm_config: Dict[str, Any]) -> str:
    """
    Extract model name based on provider from LLM configuration.

    Args:
        provider: LLM provider name (openai, anthropic, gemini, vllm)
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
    elif provider == "vllm":
        raise NotImplementedError("vLLM provider is currently not implemented")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return model_name