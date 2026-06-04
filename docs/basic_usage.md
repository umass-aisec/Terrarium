---
title: Basic Usage
---

# Basic Usage

## Multi-Agent Systems

Before going into details, we'll first convey what Terrarium does. As LLM-based agents explore the world through APIs--exploring the internet, connecting with other agents, and interacting with human users--we need a simple framework and suite of environments to evaluate these agents in benign or adversairal settings. Terrarium is made for researchers and practitioners who want to study or evaluate multi-agent systems composed of LLM-based agents in divese scenarios for safety, security, evals, or emergent behaviors.

## Why Terrarium?

Agents wrap an LLM client, receive observations, decide when to call tools, and write their trajectories to the configured loggers.

## Blackboards

A blackboard is an append-only communication and event log. It is the shared surface agents use to communicate with each other and inspect prior events.

## Communication Protocols

Protocols decide how agents take turns. The default two-phase protocol separates planning from execution so agents can coordinate before taking environment actions.

## Environments

Environments define task state, available tools, and evaluation logic. New environments should inherit from `terrarium.environments.abstract_environment.AbstractEnvironment`.

## Tools

Terrarium tools are discovered from the active environment and phase. Blackboard tools are protocol-level communication tools, while environment tools mutate or query task state.
