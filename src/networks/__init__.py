"""Communication network topologies (NetworkX-backed).

Networks in this package define *who can communicate with whom* by producing a
graph whose edges map to blackboard memberships (i.e., one blackboard per edge).
"""

from .base import CommunicationNetwork
from .complete import CompleteNetwork
from .factory import build_communication_network
from .path import PathNetwork
from .random import RandomNetwork
from .star import StarNetwork

__all__ = [
    "CommunicationNetwork",
    "CompleteNetwork",
    "build_communication_network",
    "PathNetwork",
    "RandomNetwork",
    "StarNetwork",
]
