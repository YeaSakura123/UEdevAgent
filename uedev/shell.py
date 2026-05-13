from __future__ import annotations

import sys

from .tools import shell as _shell

sys.modules[__name__] = _shell
