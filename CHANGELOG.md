# Changelog

All notable changes to this project are documented in this file.

## Next Changes
- Optimize environment decomposition to be more intuitive
- Add more sophisicated examples
- Update README.md

## [Unreleased]

### Added
- `IsThisSeatTakenEnvironment`: a social seat-selection environment generated in-process from the config and seed, with no CoLLAB dependency. See `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_environment.md`.
- `terrarium/compaction/`: compaction of blackboard history in agent prompts, enabled per run by an `llm.compaction` block. See `terrarium/compaction/README.md`.
- `terrarium/personas/`: Big Five persona shaping through prompting. See `terrarium/personas/README.md`.
- Agent-created private channels: blackboard tool `create_channel`, offered in environments whose Tools class sets `supports_private_channels = True`.
- Blackboard tool `recall`, offered when `llm.compaction.retrieval_enabled` is true.
- `examples/base_main.py` flags `--seed`, `--persona`, `--personas`, `--model` and `--compaction-model`, which override the loaded config.

### Changed
- `SequentialCommunicationProtocol` compacts blackboard history only when `llm.compaction` is configured; without it, prompts are unchanged. In environments that support private channels, each channel's history is prefixed with `[channel <id> — with <participants>]` so agents can address a channel by id.
- `ToolsetDiscovery.get_blackboard_tool_names()` and `get_tools_for_blackboard()` take optional `environment_name` and `retrieval_enabled` arguments. Calls without them return the same tools as before.
- `BaseAgent` accepts `retrieval_enabled` (default `False`); `build_agents` sets it from `llm.compaction.retrieval_enabled`.
- `Megaboard` adds `create_channel()`, `is_private_channel()` and `recall()`.
- `OpenAIClient` treats `gpt-5.5` as temperature-restricted, so temperature is not sent for it.
- `IsThisSeatTakenEnvironment` is added to the environment registry and the `terrarium.environments` namespaces.

## [v0.2.0] - 2026-02-21

### Changed
- Reorganized internal package structure to separate core runtime components and tool-related modules:
  - Added `terrarium/core/` for runtime primitives (`async_utils.py`, `blackboard.py`, `logger.py`)
  - Added `terrarium/tools/` for tool plumbing (`environment.py`, `discovery.py`, `prompts.py`)
- Updated internal imports, examples, and tests to use the new module paths.

- Removed legacy root-level modules as part of a clean break:
  - `terrarium/async_utils.py`
  - `terrarium/blackboard.py`
  - `terrarium/logger.py`
  - `terrarium/environment_tools.py`
  - `terrarium/toolset_discovery.py`
  - `terrarium/tool_prompt_utils.py`

- Import path migration:
  - `from terrarium.blackboard import ...` -> `from terrarium.core.blackboard import ...`
  - `from terrarium.logger import ...` -> `from terrarium.core.logger import ...`
  - `from terrarium.async_utils import ...` -> `from terrarium.core.async_utils import ...`
  - `from terrarium.environment_tools import ...` -> `from terrarium.tools.environment import ...`
  - `from terrarium.toolset_discovery import ...` -> `from terrarium.tools.discovery import ...`
  - `from terrarium.tool_prompt_utils import ...` -> `from terrarium.tools.prompts import ...`

## [0.1.1] - 2026-02-21

### Added
- Initial Terrarium release.
