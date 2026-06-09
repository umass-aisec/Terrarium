# Changelog

All notable changes to this project are documented in this file.

## Next Changes
- Optimize environment decomposition to be more intuitive
- Add more sophisicated examples
- Update README.md
- Add GitHub Actions documentation deployment to build Sphinx docs and publish them to `gh-pages`.
- Publish `main` documentation to `latest/` and release tag documentation to versioned paths such as `v0.1.2/`.
- Add a generated docs landing page with a version list and redirect to the default documentation version.
- Make the Sphinx `html_baseurl` configurable through `TERRARIUM_DOCS_BASEURL` for versioned canonical URLs.

## [v0.1.2] - 2026-06-08

### Added
- Added the Sphinx/Furo documentation site with quick start, basic usage, component guides, API reference pages, and framework comparison docs.
- Added or expanded provider support for OpenAI, Microsoft Foundry, Grok, Anthropic, Gemini, Together, Fireworks, and local vLLM.
- Added configurable external MCP server support for provider clients.
- Added compaction utilities for shortening long blackboard event histories.
- Added tests for provider clients, external MCP routing, network factory behavior, local protocol behavior, and collusion-related scenarios.

### Changed
- Reorganized core runtime primitives under `terrarium.core`.
- Reorganized tool plumbing under `terrarium.tools`.
- Reorganized LLM provider clients and vLLM runtime helpers under `terrarium.llm`.
- Updated examples and configs for shipped environments and attack scenarios.

### Removed
- Removed deprecated server/MCP paths in favor of in-process environment and blackboard tools plus optional external MCP servers.

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
