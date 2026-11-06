"""Test isolation: server tests write their command record to a temporary folder, never to workspace/."""
import os
import sys

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RECON = os.path.join(ROOT_DIR, "reconstructed")
if _RECON not in sys.path:
    sys.path.insert(0, _RECON)


@pytest.fixture(autouse=True)
def _isolated_command_log(tmp_path):
    try:
        from pyjamovi.server import STATE
        from pyjamovi.command_log import CommandLog
    except Exception:          # server dependencies missing: nothing to isolate
        yield
        return
    saved = (STATE.cmdlog, STATE.log_sig)
    STATE.cmdlog = CommandLog(str(tmp_path / "workspace_log"))
    STATE.log_sig = None
    try:
        yield
    finally:
        STATE.cmdlog, STATE.log_sig = saved
