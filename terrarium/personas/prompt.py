"""Wraps a persona's trait clause around a task, following the paper's prompt format."""

from terrarium.personas.persona import Persona, build_trait_clause


def build_persona_description(persona: Persona) -> str:
    """Combine the persona's role/context description with its trait clause."""
    parts = [part for part in (persona.description, build_trait_clause(persona)) if part]
    return " ".join(parts)


def build_persona_prompt(persona: Persona, task: str) -> str:
    """
    Wrap `task` with a persona-shaping instruction:

        For the following task, respond in a way that matches this
        description: "{persona description}. I'm {qualified adjectives}."

        {task}
    """
    description = build_persona_description(persona)
    if not description:
        return task

    instruction = f'For the following task, respond in a way that matches this description: "{description}"'
    return f"{instruction}\n\n{task}"
