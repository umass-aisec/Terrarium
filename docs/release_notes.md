---
title: Release Notes
---

# Release Notes

## v0.2.1 - 2026-10-23

Terrarium v0.2.1 is a bug-fix release that restores how long an agent's turn lasts.

### Fixed

- In v0.2.0, any successful environment tool call ended the agent's turn, including read-only lookups such as the Hospital environment's `get_job_queue`. An agent that checked its queue could not act on what it saw in the same turn. A turn now ends only after a committed action, as in v0.1.x.

### Upgrading

- Environment tools signal a committed action with `"action_committed": True` in their result; the shipped DCOP environments already do. Add it to custom tools whose calls should end the agent's turn, and leave it out of read-only tools. `BaseAgent` strips the key, so models never see it.

## v0.2.0 - 2026-06-08

Terrarium v0.2.0 focuses on making the project easier to install, understand, and extend. The release adds a proper documentation site, clearer examples, broader provider support, and more explicit runtime modules for agents, tools, environments, and logs.

### Documentation

- Added a Sphinx documentation site using MyST Markdown and the Furo theme.
- Added a first-run guide, configuration guide, and local docs build instructions.
- Added component guides for agents, environments, communication, networks, tools, inference clients, attacks, and logging.
- Added generated API reference pages for the main Terrarium modules.
- Added a framework comparison page explaining where Terrarium differs from orchestration frameworks and fixed benchmarks.
- Added GitHub Pages deployment for versioned docs: `main` publishes to `latest/`, release tags publish to paths such as `v0.2.0/`, and the root page links available versions.

### Runtime Structure

- Reorganized core runtime primitives into `terrarium.core`, including blackboards, loggers, and async helpers.
- Reorganized tool plumbing into `terrarium.tools`, including tool discovery, environment tool resolution, and vLLM tool prompt helpers.
- Reorganized provider clients under `terrarium.llm` and exposed local vLLM runtime helpers.
- Added compaction utilities for shortening long blackboard event histories before prompting.

### Inference and Tools

- Added or expanded support for OpenAI, Microsoft Foundry, Grok, Anthropic, Gemini, Together, Fireworks, and local vLLM clients.
- Added configurable external MCP server support so provider clients can discover and call external tools.
- Added `.env.example` coverage for supported providers and a Foundry setup guide.

### Environments and Examples

- Updated example configs for Meeting Scheduling, Personal Assistant, Smart Grid, Hospital, and Jira Ticket environments.
- Added clearer attack configuration and updated the attack runner for agent poisoning, context overflow, and protocol-level poisoning experiments.
- Added communication-network configuration options for deterministic and random graph topologies.

### Testing and Reliability

- Added tests for provider clients, external MCP tool routing, tool-name normalization, network factory behavior, local protocol behavior, and collusion-related scenarios.
- Made experiment-dependent tests skip cleanly when the optional experiment sources are not checked out.
- Removed deprecated server/MCP paths in favor of in-process environment and blackboard tools plus optional external MCP servers.

The full changelog is maintained in `CHANGELOG.md` at the repository root.
