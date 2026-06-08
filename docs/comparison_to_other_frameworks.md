---
title: Comparison to Other Frameworks
---

# Comparison to Other Frameworks

## How Terrarium Differs from Other Multi-Agent Frameworks

Terrarium is not primarily a production agent orchestrator, software-development automation system, debate protocol, planning stack, or fixed benchmark. **It is a configurable environment and communication laboratory for studying decentralized LLM-agent coordination, safety, privacy, and security.**

- [OpenHands](https://github.com/OpenHands/OpenHands): AI software developer agents that use code, shell, browser, and sandboxed runtime tools. <!-- ~76k stars -->
  <div style="color: #2e7d32;">+ Whereas OpenHands targets autonomous software development in real computer-use environments, Terrarium offers controlled multi-agent environments with explicit rewards, private information, communication topology, and adversarial conditions beyond software-development workspaces.</div>
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT): role-driven software-company workflows built around SOPs and specialized agents. <!-- ~68.6k stars -->
  <div style="color: #2e7d32;">+ Whereas MetaGPT instantiates agents for software-company workflows, Terrarium offers domain-agnostic environment construction and configurable communication protocols that are not tied to a fixed software-company process.</div>
- [AutoGen](https://github.com/microsoft/autogen): conversational multi-agent orchestration; the repository now points new users toward Microsoft Agent Framework. <!-- ~58.7k stars -->
  <div style="color: #2e7d32;">+ Whereas AutoGen emphasizes conversational agent orchestration, Terrarium offers a blackboard-centered communication layer that is decoupled from agent roles, plus explicit topology and security-evaluation controls.</div>
- [CrewAI](https://github.com/crewAIInc/crewAI): role, task, crew, and flow abstractions for production-style agent automation. <!-- ~52.9k stars -->
  <div style="color: #2e7d32;">+ Whereas CrewAI organizes agent behavior through roles, tasks, crews, and flows, Terrarium allows more fine-grained control by letting users directly configure which agents can communicate, what information they can see, and how messages move through the environment.</div>
- [LangGraph](https://github.com/langchain-ai/langgraph): graph-based orchestration for durable, stateful, long-running agents. <!-- ~34k stars -->
  <div style="color: #2e7d32;">+ Whereas LangGraph models agent applications as durable workflow graphs, Terrarium offers decentralized environment simulation with measurable utility, private observations, protocol control, and adversarial interventions.</div>
- [ChatDev](https://github.com/OpenBMB/ChatDev): a legacy virtual software-company system that has evolved into a broader zero-code multi-agent orchestration platform. <!-- ~33.3k stars -->
  <div style="color: #2e7d32;">+ Whereas ChatDev centers on virtual software-company and zero-code orchestration workflows, Terrarium offers lower-level experimental control over environment state, reward, communication access, and attack conditions.</div>
- [CAMEL](https://github.com/camel-ai/camel): role-playing agents, agent societies, synthetic data, memory, tools, and benchmark support. <!-- ~17.1k stars -->
  <div style="color: #2e7d32;">+ Whereas CAMEL focuses on role-playing agents, agent societies, and synthetic data, Terrarium offers explicit cooperative environment definitions, private information boundaries, and configurable communication topology for coordination studies.</div>
- [AgentVerse](https://github.com/OpenBMB/AgentVerse): task-solving and simulation frameworks for multi-agent applications. <!-- ~5k stars -->
  <div style="color: #2e7d32;">+ Whereas AgentVerse provides task-solving and simulation frameworks for multi-agent applications, Terrarium offers finer-grained protocol, blackboard, and adversarial-evaluation controls for fixed but configurable multi-agent experiments.</div>
- [Multi-Agent Debate / MAD](https://github.com/Skytliang/Multi-Agents-Debate): debate-style reasoning with agents and judges. <!-- ~578 stars -->
  <div style="color: #2e7d32;">+ Whereas multi-agent debate systems center on fixed critique-and-judge interaction patterns, Terrarium offers debate-like interaction as one possible protocol among many, with arbitrary topology, tools, environment dynamics, and attack modules.</div>
- [X-Teaming](https://github.com/salman-lui/x-teaming): adaptive multi-agent red-teaming for multi-turn jailbreak discovery. <!-- ~67 stars -->
  <div style="color: #2e7d32;">+ Whereas X-Teaming specializes in multi-agent jailbreak discovery, Terrarium offers a general MAS sandbox for both attack and defense studies across non-jailbreak settings such as coordination, privacy, and tool misuse.</div>
- [AgentLeak](https://github.com/Privatris/AgentLeak): privacy-leakage auditing across multi-agent communication channels. <!-- ~21 stars -->
  <div style="color: #2e7d32;">+ Whereas AgentLeak audits leakage across existing multi-agent communication channels, Terrarium offers the ability to design the environment, communication topology, and protocol being audited, not only measure leakage in predefined settings.</div>
- [AgentLAB](https://github.com/TanqiuJiang/AgentLAB): long-horizon attack benchmark with multi-turn attack families and tool-enabled environments. <!-- ~19 stars -->
  <div style="color: #2e7d32;">+ Whereas AgentLAB provides a long-horizon attack benchmark, Terrarium offers custom environment construction and communication-control primitives for studying why those attacks succeed or fail.</div>
- [TAMAS](https://github.com/microsoft/TAMAS): adversarial benchmark for testing AutoGen and CrewAI-style multi-agent systems. <!-- ~12 stars -->
  <div style="color: #2e7d32;">+ Whereas TAMAS tests existing AutoGen and CrewAI-style systems with predefined adversarial cases, Terrarium offers extensible environment, protocol, and attack-module authoring rather than only running a fixed benchmark suite.</div>
- [CalBench](https://arxiv.org/abs/2605.09823): a calendar-scheduling benchmark for studying coordination and privacy tradeoffs.
  <div style="color: #2e7d32;">+ Whereas CalBench standardizes one calendar-scheduling coordination benchmark, Terrarium offers the general environment-building tools to create CalBench-style tasks and adapt them to new domains, protocols, and attacks.</div>
- [ATLAS](https://arxiv.org/abs/2509.25586): constraint-aware multi-agent planning for real-world travel planning.
  <div style="color: #2e7d32;">+ Whereas ATLAS optimizes constraint-aware planning with specialized planner, critic, and search roles, Terrarium offers domain-agnostic primitives for building constraint-heavy tasks without requiring a fixed constraint/planner/critic/searcher architecture.</div>
