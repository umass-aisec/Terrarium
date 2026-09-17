"""
Persona trait shaping via prompting, per Jiang et al., "Personality Traits in
Large Language Models". Personality is controlled purely through natural
language (no fine-tuning): each Big Five dimension is set to a level from 1-9
and rendered as a qualified first-person trait clause appended to a persona
description.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from terrarium.personas.adjectives import (
    VERBOSITY_COMPACT,
    VERBOSITIES,
    get_domain_adjectives,
    get_facet_adjectives,
    is_domain,
)

MIN_LEVEL = 1
MAX_LEVEL = 9
NEUTRAL_LEVEL = 5

# Level -> qualifier prefixed to the low adjective (levels 1-4) or high
# adjective (levels 6-9). Level 5 is neutral and contributes no qualifier.
_LOW_QUALIFIERS = {1: "extremely", 2: "very", 3: None, 4: "a bit"}
_HIGH_QUALIFIERS = {6: "a bit", 7: None, 8: "very", 9: "extremely"}


def qualify(level: int, low_adjective: str, high_adjective: str) -> Optional[str]:
    """
    Render one adjective pair at the given intensity level (1-9).

    Returns None at the neutral level (5), since it contributes nothing to
    the trait clause.
    """
    if not MIN_LEVEL <= level <= MAX_LEVEL:
        raise ValueError(f"Level must be between {MIN_LEVEL} and {MAX_LEVEL}, got {level}.")

    if level == NEUTRAL_LEVEL:
        return None
    if level < NEUTRAL_LEVEL:
        adjective, qualifier = low_adjective, _LOW_QUALIFIERS[level]
    else:
        adjective, qualifier = high_adjective, _HIGH_QUALIFIERS[level]

    return f"{qualifier} {adjective}" if qualifier else adjective


def _join_adjectives(adjectives: List[str]) -> str:
    if len(adjectives) == 1:
        return adjectives[0]
    if len(adjectives) == 2:
        return f"{adjectives[0]} and {adjectives[1]}"
    return ", ".join(adjectives[:-1]) + f", and {adjectives[-1]}"


@dataclass
class Persona:
    """
    A Big Five personality specification.

    `traits` maps either a domain code (EXT, AGR, CON, NEU, OPE — shapes
    every facet under that domain) or a facet code/name (e.g. "A1" or
    "Trust" — shapes just that facet) to an intensity level from 1-9.
    Levels of 5 (neutral) are allowed but contribute nothing to the prompt.

    `verbosity` controls how domain-level traits are expanded into
    adjectives: "compact" (default) uses a fixed 4-5 word subset per domain
    for shorter, reproducible prompts; "full" uses every Table 12 facet
    marker, reproducing the paper's complete prompting strategy. It has no
    effect on facet-level traits, which are already a single row.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    traits: Dict[str, int] = field(default_factory=dict)
    verbosity: str = VERBOSITY_COMPACT

    def __post_init__(self) -> None:
        if self.verbosity not in VERBOSITIES:
            raise ValueError(f"Unknown verbosity '{self.verbosity}'. Expected one of {VERBOSITIES}.")

    @classmethod
    def from_big_five(
        cls,
        ext: int = NEUTRAL_LEVEL,
        agr: int = NEUTRAL_LEVEL,
        con: int = NEUTRAL_LEVEL,
        neu: int = NEUTRAL_LEVEL,
        ope: int = NEUTRAL_LEVEL,
        name: Optional[str] = None,
        description: Optional[str] = None,
        verbosity: str = VERBOSITY_COMPACT,
    ) -> "Persona":
        """Build a Persona from the standard 5-tuple of domain-wide levels."""
        return cls(
            name=name,
            description=description,
            traits={"EXT": ext, "AGR": agr, "CON": con, "NEU": neu, "OPE": ope},
            verbosity=verbosity,
        )


def build_trait_clause(persona: Persona) -> str:
    """
    Render a persona's traits as a first-person self-description, e.g.
    "I'm a bit introverted, a bit unenergetic, ... and a bit unadventurous."

    Returns "" if every trait is at the neutral level (5) or no traits are set.
    """
    adjectives: List[str] = []
    for key, level in persona.traits.items():
        pairs = (
            get_domain_adjectives(key, persona.verbosity)
            if is_domain(key)
            else get_facet_adjectives(key)
        )
        for low, high in pairs:
            rendered = qualify(level, low, high)
            if rendered:
                adjectives.append(rendered)

    if not adjectives:
        return ""
    return f"I'm {_join_adjectives(adjectives)}."
