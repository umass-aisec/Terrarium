"""Communication network topologies (NetworkX-backed).

Networks in this package define *who can communicate with whom* by producing a
graph whose edges map to blackboard memberships (i.e., one blackboard per edge).
"""

from .base import CommunicationNetwork
from .barabasi_albert import BarabasiAlbertNetwork
from .complete import CompleteNetwork
from .erdos_renyi import ErdosRenyiNetwork
from .factory import build_communication_network
from .path import PathNetwork
from .star import StarNetwork
from .watts_strogatz import WattsStrogatzNetwork

__all__ = [
    "BarabasiAlbertNetwork",
    "CommunicationNetwork",
    "CompleteNetwork",
    "ErdosRenyiNetwork",
    "build_communication_network",
    "PathNetwork",
    "StarNetwork",
    "WattsStrogatzNetwork",
]
