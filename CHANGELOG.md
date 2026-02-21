# Changelog

All notable changes to this project are documented in this file.

## Next Changes
- Optimize environment decomposition to be more intuitive
- Add more sophisicated examples
- Update README.md

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
