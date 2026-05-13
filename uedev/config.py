from __future__ import annotations

import sys

from .state import config as _config

sys.modules[__name__] = _config
