---
title: Attack Scenarios
---

# Attack Scenarios

Terrarium implements a few attack scenarios that can be mixed into simulations with `examples/attack_main.py`.

## Agent Poisoning

Replaces every `post_message` payload from a compromised agent before it reaches the blackboard.

```bash
python examples/attack_main.py \
  --config examples/configs/meeting_scheduling.yaml \
  --poison_payload examples/configs/attack_config.yaml \
  --attack_type agent_poisoning
```

## Context Overflow

Appends a large filler block to agent messages to pressure downstream context management.

```bash
python examples/attack_main.py \
  --config examples/configs/meeting_scheduling.yaml \
  --poison_payload examples/configs/attack_config.yaml \
  --attack_type context_overflow
```

## Communication Protocol Poisoning

Injects malicious system messages into every blackboard through the communication layer.

```bash
python examples/attack_main.py \
  --config examples/configs/meeting_scheduling.yaml \
  --poison_payload examples/configs/attack_config.yaml \
  --communication_protocol_poisoning
```
