"""
msm_sensitivity
Tests msm sensitivity to hyperparameters
"""

# Add imports here
from .msm_sensitivity import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
