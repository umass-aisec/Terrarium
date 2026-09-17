# Changelog

All notable changes to this project are documented in this file.

## Next Changes
- Optimize environment decomposition to be more intuitive
- Add more sophisicated examples
- Update README.md

## [Unreleased]

### Added
- `IsThisSeatTakenEnvironment`: a social seat-selection environment generated in-process from the config and seed, with no CoLLAB dependency. See `terrarium/environments/dcops/is_this_seat_taken/is_this_seat_taken_environment.md`.
- `terrarium/personas/`: Big Five persona shaping through prompting. See `terrarium/personas/README.md`.
- Agent-created private channels: blackboard tool `create_channel`, offered in environments whose Tools class sets `supports_private_channels = True`.
- Blackboard tool `recall`, offered when `llm.compaction.retrieval_enabled` is true.
- `examples/base_main.py` flags `--seed`, `--persona`, `--personas`, `--model` and `--compaction-model`, which override the loaded config.

### Changed
- Reworked the compaction utilities added in v0.2.0 into six interchangeable mechanisms (`baseline`, `anchored`, `eviction`, `extractive`, `query_conditioned`, `structured`) with optional context-event pinning and a `CompactionLogger`. Summary requests use the `params` from the `llm.compaction` provider block, with `max_tokens` defaulting to 500. See `terrarium/compaction/README.md`.
- `SequentialCommunicationProtocol` compacts blackboard history only when `llm.compaction` is configured; without it, prompts are unchanged. In environments that support private channels, each channel's history is prefixed with `[channel <id> — with <participants>]` so agents can address a channel by id.
- `ToolsetDiscovery.get_blackboard_tool_names()` and `get_tools_for_blackboard()` take optional `environment_name` and `retrieval_enabled` arguments. Calls without them return the same tools as before.
- `BaseAgent` accepts `retrieval_enabled` (default `False`); `build_agents` sets it from `llm.compaction.retrieval_enabled`.
- `Megaboard` adds `create_channel()`, `is_private_channel()` and `recall()`.
- `OpenAIClient` treats `gpt-5.5` as temperature-restricted, so temperature is not sent for it.
- `IsThisSeatTakenEnvironment` is added to the environment registry and the `terrarium.environments` namespaces.

## [v0.2.0] - 2026-06-08

### Added
- Added the Sphinx/Furo documentation site with quick start, basic usage, component guides, API reference pages, and framework comparison docs.
- Added GitHub Actions documentation deployment to build Sphinx docs and publish them to `gh-pages`.
- Added versioned docs publishing: `main` documentation deploys to `latest/`, while release tags deploy to versioned paths such as `v0.2.0/`.
- Added a generated docs landing page with a version list and redirect to the default documentation version.
- Added or expanded provider support for OpenAI, Microsoft Foundry, Grok, Anthropic, Gemini, Together, Fireworks, and local vLLM.
- Added configurable external MCP server support for provider clients.
- Added compaction utilities for shortening long blackboard event histories.
- Added tests for provider clients, external MCP routing, network factory behavior, local protocol behavior, and collusion-related scenarios.

### Changed
- Reorganized core runtime primitives under `terrarium.core`.
- Reorganized tool plumbing under `terrarium.tools`.
- Reorganized LLM provider clients and vLLM runtime helpers under `terrarium.llm`.
- Made the Sphinx `html_baseurl` configurable through `TERRARIUM_DOCS_BASEURL` for versioned canonical URLs.
- Made experiment-dependent tests skip cleanly when the optional experiment sources are not checked out.
- Updated examples and configs for shipped environments and attack scenarios.

### Removed
- Removed deprecated server/MCP paths in favor of in-process environment and blackboard tools plus optional external MCP servers.

## [0.1.1] - 2026-02-21

### Added
- Initial Terrarium release.
