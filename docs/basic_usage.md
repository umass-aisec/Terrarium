---
title: Basic Usage
---

# Basic Usage

## Why Terrarium?

Before going into details, we'll first convey what Terrarium does. As LLM-based agents explore the world through APIs--exploring the internet, connecting with other agents, and interacting with human users--we need a simple framework and suite of environments to evaluate these agents in benign or adversarial settings. Terrarium focuses on **cooperative environments** where agents share the same reward function. Terrarium is **made for researchers and practitioners** who want to study or evaluate multi-agent systems composed of LLM-based agents in diverse scenarios for **safety, security, evals, or emergent behaviors**. Unlike other frameworks, we explicitly ground our environments to multi-agent decision-making problem formulations that have been thoroughly studied over the decades.

Our framework is modular, allowing you to swap in and out variations of the following components:
- Agents (`BaseAgent`)
- Environment (`AbstractEnvironment`)
- Communication Protocols (`BaseCommunicationProtocol`)
- Communication Networks (`CommunicationNetwork`)
- Communication Channel Representation (`Blackboard`,)

## Environments

### Single-Step
We ground our single-step environments as Distributed Constraint Optimization Problems ([DCOPs](https://arxiv.org/pdf/1602.06347)) where agents have partial-observations of the world around them and must coordinate with other agents, decentrally, to optimize a cooperative objective.

### Multi-Step
Work in Progress.
<!-- > [!NOTE]
> We  -->
