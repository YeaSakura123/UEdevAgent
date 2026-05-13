from __future__ import annotations

import sys

from .runtime import agent as _agent

sys.modules[__name__] = _agent
