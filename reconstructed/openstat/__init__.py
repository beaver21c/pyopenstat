"""
OpenStat Studio (`openstat`) — Standalone MIT-Licensed Open-Source Statistical Engine & Claude Code Runtime.
Provides 28 statistical analyses (including Panel FE/RE/Hausman & CFA/SEM), customizable publication plots (`PlotStyle`),
10-format analyzed table export (`.xlsx`, `.docx`, `.hwpx`, `.tex`, `.csv`, `.tsv`, `.md`, `.html`, `.json`, `.omv`),
and live bidirectional `.py` syntax authoring.

Analytical models, menu taxonomy, and `.omv` format interoperability designed with reference to jamovi (https://www.jamovi.org).
"""

from pyjamovi import *  # noqa: F401,F403
from pyjamovi import __all__, __version__  # noqa: F401
