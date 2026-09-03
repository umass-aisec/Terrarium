"""Example personas."""

from terrarium.personas.persona import Persona

# name -> (Big Five levels, description). Unspecified domains stay neutral.
_SPECS = {
    "cautious skeptic": (dict(ext=3, agr=2, con=8), "A cautious skeptic."),
    "anxious pushover": (dict(agr=8, neu=8), "An anxious pushover."),
    "confident collaborator": (dict(ext=7, agr=7, con=5), "A confident collaborator."),
    "diplomat": (dict(ext=9, agr=9), "A diplomat."),
    "territorial": (dict(ext=1, agr=1), "A territorial holdout."),
}

PRESETS = {
    name: Persona.from_big_five(name=name, description=description, **levels)
    for name, (levels, description) in _SPECS.items()
}

CAUTIOUS_SKEPTIC = PRESETS["cautious skeptic"]
ANXIOUS_PUSHOVER = PRESETS["anxious pushover"]
CONFIDENT_COLLABORATOR = PRESETS["confident collaborator"]
DIPLOMAT = PRESETS["diplomat"]
TERRITORIAL = PRESETS["territorial"]
