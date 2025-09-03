r"""
    This package contains utils for jax.
"""

import jax   # installation cpu version: https://docs.jax.dev/en/latest/installation.html
jax.config.update("jax_enable_x64", True)

from .model import *
from .utils import *
from .fit_vertex import *
