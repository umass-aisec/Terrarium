# Changelog

All notable changes to this project are documented in this file.

## Next Changes
- Optimize environment decomposition to be more intuitive
- Add more sophisicated examples
- Update README.md

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
