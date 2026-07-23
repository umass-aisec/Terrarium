"""Big Five persona shaping via prompting (no fine-tuning required)."""

from terrarium.personas.adjectives import (
    COMPACT_TRAITS,
    DOMAINS,
    DOMAIN_NAMES,
    TABLE_12,
    VERBOSITY_COMPACT,
    VERBOSITY_FULL,
    get_domain_adjectives,
    get_facet_adjectives,
)
from terrarium.personas.persona import (
    MAX_LEVEL,
    MIN_LEVEL,
    NEUTRAL_LEVEL,
    Persona,
    build_trait_clause,
    qualify,
)
from terrarium.personas.prompt import build_persona_description, build_persona_prompt
from terrarium.personas.presets import (
    ANXIOUS_PUSHOVER,
    CAUTIOUS_SKEPTIC,
    CONFIDENT_COLLABORATOR,
    DIPLOMAT,
    PRESETS,
    TERRITORIAL,
)

__all__ = [
    "DOMAINS",
    "DOMAIN_NAMES",
    "TABLE_12",
    "COMPACT_TRAITS",
    "VERBOSITY_COMPACT",
    "VERBOSITY_FULL",
    "get_domain_adjectives",
    "get_facet_adjectives",
    "MIN_LEVEL",
    "MAX_LEVEL",
    "NEUTRAL_LEVEL",
    "Persona",
    "build_trait_clause",
    "qualify",
    "build_persona_description",
    "build_persona_prompt",
    "CAUTIOUS_SKEPTIC",
    "ANXIOUS_PUSHOVER",
    "CONFIDENT_COLLABORATOR",
    "DIPLOMAT",
    "TERRITORIAL",
    "PRESETS",
]
