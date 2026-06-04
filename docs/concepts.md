---
title: Core Concepts
---

# Core Concepts

Terrarium separates a multi-agent simulation into replaceable pieces.

## Agents

Agents wrap an LLM client, receive observations, decide when to call tools, and write their trajectories to the configured loggers.

## Blackboards

A blackboard is an append-only communication and event log. It is the shared surface agents use to communicate with each other and inspect prior events.

## Communication Protocols

Protocols decide how agents take turns. The default two-phase protocol separates planning from execution so agents can coordinate before taking environment actions.

## Environments

Environments define task state, available tools, and evaluation logic. New environments should inherit from `terrarium.environments.abstract_environment.AbstractEnvironment`.

## Tools

Terrarium tools are discovered from the active environment and phase. Blackboard tools are protocol-level communication tools, while environment tools mutate or query task state.
