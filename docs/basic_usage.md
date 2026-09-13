---
title: Basic Usage
---

# Basic Usage

## Why Terrarium?

Before going into details, we'll first convey what Terrarium does. As LLM-based agents explore the world through APIs--exploring the internet, connecting with other agents, and interacting with human users--we need a simple framework and suite of environments to evaluate these agents in benign or adversarial settings. Terrarium focuses on **cooperative environments** where agents share the same reward function. Terrarium is **made for researchers and practitioners** who want to study or evaluate multi-agent systems composed of LLM-based agents in diverse scenarios for **safety, security, evals, or emergent behaviors**. Unlike other frameworks, we explicitly ground our environments to multi-agent decision-making problem formulations.

```{admonition} TL;DR
:class: important

Terrarium is a simple, modular framework meant to be customized and hacked for researchers and practitioners simulating diverse multi-agent scenarios in cooperative settings.
```

### How Terrarium Differs from Other Multi-Agent Frameworks

Most multi-agent frameworks emphasize production orchestration, software-development automation, role-playing, debate, planning, or fixed benchmark evaluation. Terrarium differs from each of these frameworks, providing more useful and extensible features for studying multi-agent systems in more general settings and allowing the user to easily swap in and out components of the multi-agent system .

For the full framework-by-framework comparison, see [Comparison to Other Frameworks](comparison_to_other_frameworks.md#how-terrarium-differs-from-other-multi-agent-frameworks).

## The Framework

Our framework is modular, allowing you to swap in and out variations of the following components for your particular simulation:
- Agents (`BaseAgent`)
- Environment (`AbstractEnvironment`)
- Communication Protocols (`BaseCommunicationProtocol`)
- Communication Networks (`CommunicationNetwork`)
- Communication Channel Representation (`Blackboard`)

## Environments

### Single-Step
We ground our single-step environments as Distributed Constraint Optimization Problems ([DCOPs](https://arxiv.org/pdf/1602.06347)) where agents have partial-observations of the world around them and must coordinate with other agents, decentrally, to optimize a cooperative objective.

Terrarium currently includes the following single-step environments:

- `MeetingSchedulingEnvironment`: Calendar agents coordinate meeting attendance across overlapping schedules, preferences, and constraints.
- `PersonalAssistantEnvironment`: Assistant agents choose outfits while balancing personal preferences, social norms, and constrained item availability.
- `SmartGridEnvironment`: Home agents coordinate appliance or machine assignments to shared renewable energy sources while avoiding grid overload.
- `HospitalEnvironment`: Hospital service agents coordinate patient workflow scheduling and scarce resource transfers across healthcare departments.
- `JiraTicketEnvironment`: Software agents assign issue microtasks while balancing skills, availability, priorities, and workload constraints.

### Multi-Step
Although our single-step environments can be generalized to the multi-step setting, we are actively designing new multi-step environments with richer and more realistic action and state spaces.

```{warning}
Coming Soon!
```
