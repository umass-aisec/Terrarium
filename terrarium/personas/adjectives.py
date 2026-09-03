"""
Big Five trait-adjective database.

Transcribed from Table 12 (Appendix) of Jiang et al., "Personality Traits in
Large Language Models" (https://www.researchgate.net/publication/372074802),
itself adapted from Goldberg's bipolar trait markers. Each row pairs a facet
of one of the five domains with the adjective that marks its low end and the
adjective that marks its high end.

A few rows are tagged with the bare domain code instead of a facet (e.g.
"AGR" rather than "A1"): these are higher-order markers from the same table
that the paper doesn't tie to a specific facet.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

EXT = "EXT"
AGR = "AGR"
CON = "CON"
NEU = "NEU"
OPE = "OPE"

DOMAINS: Tuple[str, ...] = (EXT, AGR, CON, NEU, OPE)

@dataclass(frozen=True)
class FacetMarker:
    domain: str
    facet_code: str
    facet_name: str
    low: str
    high: str


# Table 12: Pairs of adjectival markers that map onto IPIP-NEO personality
# facets and their higher-order Big Five domains.
TABLE_12: List[FacetMarker] = [
    FacetMarker(EXT, "E1", "Friendliness", "unfriendly", "friendly"),
    FacetMarker(EXT, "E2", "Gregariousness", "introverted", "extraverted"),
    FacetMarker(EXT, "E2", "Gregariousness", "silent", "talkative"),
    FacetMarker(EXT, "E3", "Assertiveness", "timid", "bold"),
    FacetMarker(EXT, "E3", "Assertiveness", "unassertive", "assertive"),
    FacetMarker(EXT, "E4", "Activity Level", "inactive", "active"),
    FacetMarker(EXT, "E5", "Excitement-Seeking", "unenergetic", "energetic"),
    FacetMarker(EXT, "E5", "Excitement-Seeking", "unadventurous", "adventurous and daring"),
    FacetMarker(EXT, "E6", "Cheerfulness", "gloomy", "cheerful"),

    FacetMarker(AGR, "A1", "Trust", "distrustful", "trustful"),
    FacetMarker(AGR, "A2", "Morality", "immoral", "moral"),
    FacetMarker(AGR, "A2", "Morality", "dishonest", "honest"),
    FacetMarker(AGR, "A3", "Altruism", "unkind", "kind"),
    FacetMarker(AGR, "A3", "Altruism", "stingy", "generous"),
    FacetMarker(AGR, "A3", "Altruism", "unaltruistic", "altruistic"),
    FacetMarker(AGR, "A4", "Cooperation", "uncooperative", "cooperative"),
    FacetMarker(AGR, "A5", "Modesty", "self-important", "humble"),
    FacetMarker(AGR, "A6", "Sympathy", "unsympathetic", "sympathetic"),
    FacetMarker(AGR, AGR, "Agreeableness", "selfish", "unselfish"),
    FacetMarker(AGR, AGR, "Agreeableness", "disagreeable", "agreeable"),

    FacetMarker(CON, "C1", "Self-Efficacy", "unsure", "self-efficacious"),
    FacetMarker(CON, "C2", "Orderliness", "messy", "orderly"),
    FacetMarker(CON, "C3", "Dutifulness", "irresponsible", "responsible"),
    FacetMarker(CON, "C4", "Achievement-Striving", "lazy", "hardworking"),
    FacetMarker(CON, "C5", "Self-Discipline", "undisciplined", "self-disciplined"),
    FacetMarker(CON, "C6", "Cautiousness", "impractical", "practical"),
    FacetMarker(CON, "C6", "Cautiousness", "extravagant", "thrifty"),
    FacetMarker(CON, CON, "Conscientiousness", "disorganized", "organized"),
    FacetMarker(CON, CON, "Conscientiousness", "negligent", "conscientious"),
    FacetMarker(CON, CON, "Conscientiousness", "careless", "thorough"),

    FacetMarker(NEU, "N1", "Anxiety", "relaxed", "tense"),
    FacetMarker(NEU, "N1", "Anxiety", "at ease", "nervous"),
    FacetMarker(NEU, "N2", "Anger", "calm", "irritable"),
    FacetMarker(NEU, "N2", "Anger", "patient", "angry"),
    FacetMarker(NEU, "N3", "Depression", "happy", "depressed"),
    FacetMarker(NEU, "N4", "Self-Consciousness", "unselfconscious", "self-conscious"),
    FacetMarker(NEU, "N5", "Immoderation", "level-headed", "impulsive"),
    FacetMarker(NEU, "N6", "Vulnerability", "contented", "discontented"),
    FacetMarker(NEU, "N6", "Vulnerability", "emotionally stable", "emotionally unstable"),

    FacetMarker(OPE, "O1", "Imagination", "unimaginative", "imaginative"),
    FacetMarker(OPE, "O2", "Artistic Interests", "uncreative", "creative"),
    FacetMarker(OPE, "O2", "Artistic Interests", "artistically unappreciative", "artistically appreciative"),
    FacetMarker(OPE, "O3", "Emotionality", "unreflective", "reflective"),
    FacetMarker(OPE, "O3", "Emotionality", "emotionally closed", "emotionally aware"),
    FacetMarker(OPE, "O4", "Adventurousness", "uninquisitive", "curious"),
    FacetMarker(OPE, "O4", "Adventurousness", "predictable", "spontaneous"),
    FacetMarker(OPE, "O5", "Intellect", "unintelligent", "intelligent"),
    FacetMarker(OPE, "O5", "Intellect", "unanalytical", "analytical"),
    FacetMarker(OPE, "O5", "Intellect", "unsophisticated", "sophisticated"),
    FacetMarker(OPE, "O6", "Liberalism", "socially conservative", "socially progressive"),
]


VERBOSITY_COMPACT = "compact"
VERBOSITY_FULL = "full"
VERBOSITIES = (VERBOSITY_COMPACT, VERBOSITY_FULL)

# Fixed, hand-picked subset of Table 12 used for verbosity="compact". Every
# word here is a verified Table 12 marker (see TABLE_12 above) - this is a
# curation of *which* rows to keep per domain, not new adjectives. Kept
# deterministic and hardcoded (not sampled) so the same Persona always
# renders the same prompt.
#
# EXT: the paper's own worked domain-level example (Sec. 4.2.1, p.18) uses
# Gregariousness/Assertiveness/Excitement-Seeking and drops Friendliness and
# Cheerfulness; "adventurous" (not "adventurous and daring") matches that
# example's wording verbatim.
# AGR: Trust/Morality/Altruism/Cooperation/Sympathy - the same five facets
# used in this project's original persona plan, dropping Modesty to fit the
# compact budget.
# CON/NEU/OPE: the paper gives no domain-level worked example for these, so
# each picks one classic, widely-used marker per facet (the same adjectives
# that anchor these domains on standard Big Five instruments like the BFI),
# spread across distinct facets rather than clustering on one.
COMPACT_TRAITS: Dict[str, List[Tuple[str, str]]] = {
    EXT: [
        ("introverted", "extraverted"),   # E2 Gregariousness
        ("silent", "talkative"),          # E2 Gregariousness
        ("timid", "bold"),                # E3 Assertiveness
        ("unadventurous", "adventurous"), # E5 Excitement-Seeking
    ],
    AGR: [
        ("distrustful", "trustful"),      # A1 Trust
        ("immoral", "moral"),             # A2 Morality
        ("unkind", "kind"),               # A3 Altruism
        ("uncooperative", "cooperative"), # A4 Cooperation
        ("unsympathetic", "sympathetic"), # A6 Sympathy
    ],
    CON: [
        ("messy", "orderly"),                    # C2 Orderliness
        ("irresponsible", "responsible"),        # C3 Dutifulness
        ("undisciplined", "self-disciplined"),   # C5 Self-Discipline
        ("careless", "thorough"),                # CON (domain marker)
    ],
    NEU: [
        ("relaxed", "tense"),                            # N1 Anxiety
        ("calm", "irritable"),                           # N2 Anger
        ("happy", "depressed"),                          # N3 Depression
        ("emotionally stable", "emotionally unstable"),  # N6 Vulnerability
    ],
    OPE: [
        ("unimaginative", "imaginative"),   # O1 Imagination
        ("uncreative", "creative"),         # O2 Artistic Interests
        ("uninquisitive", "curious"),       # O4 Adventurousness
        ("unintelligent", "intelligent"),   # O5 Intellect
    ],
}


def get_domain_adjectives(domain: str, verbosity: str = VERBOSITY_FULL) -> List[Tuple[str, str]]:
    """
    (low, high) adjective pairs for a Big Five domain.

    verbosity="full" (default for this function) returns every facet row
    from Table 12. verbosity="compact" returns the fixed 4-5 word subset in
    COMPACT_TRAITS.
    """
    domain = domain.upper()
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Expected one of {DOMAINS}.")
    if verbosity not in VERBOSITIES:
        raise ValueError(f"Unknown verbosity '{verbosity}'. Expected one of {VERBOSITIES}.")

    if verbosity == VERBOSITY_COMPACT:
        return list(COMPACT_TRAITS[domain])
    return [(m.low, m.high) for m in TABLE_12 if m.domain == domain]


def get_facet_adjectives(facet: str) -> List[Tuple[str, str]]:
    """All (low, high) adjective pairs for a single facet, matched by code (e.g. 'A1') or name (e.g. 'Trust')."""
    facet_lower = facet.lower()
    matches = [
        (m.low, m.high)
        for m in TABLE_12
        if m.facet_code.lower() == facet_lower or m.facet_name.lower() == facet_lower
    ]
    if not matches:
        raise ValueError(f"Unknown facet '{facet}'.")
    return matches


def is_domain(key: str) -> bool:
    return key.upper() in DOMAINS
