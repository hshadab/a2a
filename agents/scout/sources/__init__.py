"""
URL Sources for Scout Agent
"""
from .base import URLSource
from .phishtank import PhishTankSource
from .openphish import OpenPhishSource
from .urlhaus import URLhausSource
from .synthetic import SyntheticSource, AlexaTopSource

__all__ = [
    'URLSource',
    'PhishTankSource',
    'OpenPhishSource',
    'URLhausSource',
    'SyntheticSource',
    'AlexaTopSource',
]
