"""Example personas."""

from terrarium.personas.persona import Persona

CAUTIOUS_SKEPTIC = Persona.from_big_five(
    ext=3,
    agr=2,
    con=8,
    name="cautious skeptic",
    description="A cautious skeptic.",
)

ANXIOUS_PUSHOVER = Persona.from_big_five(
    agr=8,
    neu=8,
    name="anxious pushover",
    description="An anxious pushover.",
)

CONFIDENT_COLLABORATOR = Persona.from_big_five(
    ext=7,
    agr=7,
    con=5,
    name="confident collaborator",
    description="A confident collaborator.",
)

PRESETS = {
    persona.name: persona
    for persona in (CAUTIOUS_SKEPTIC, ANXIOUS_PUSHOVER, CONFIDENT_COLLABORATOR)
}
