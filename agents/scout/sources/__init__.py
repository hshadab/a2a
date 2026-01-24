"""
URL Sources for Scout Agent
"""
from .base import URLSource
from .phishtank import PhishTankSource
from .openphish import OpenPhishSource
from .urlhaus import URLhausSource
from .synthetic import SyntheticSource, AlexaTopSource
from .crtsh import CertTransparencySource
from .twitter import TwitterSource
from .pastebin import PasteSiteSource

__all__ = [
    'URLSource',
    'PhishTankSource',
    'OpenPhishSource',
    'URLhausSource',
    'SyntheticSource',
    'AlexaTopSource',
    'CertTransparencySource',
    'TwitterSource',
    'PasteSiteSource',
]
