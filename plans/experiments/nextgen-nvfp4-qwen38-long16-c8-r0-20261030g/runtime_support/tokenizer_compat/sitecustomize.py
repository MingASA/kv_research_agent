"""Opt-in AIPerf 0.13.0 offline local-path compatibility shim.

Inject ONLY into AIPerf benchmark processes with environment:
  PYTHONPATH=/workspace/inputs/aiperf/aiperf_v3_tokenizer_compat
  CJYU_AIPERF_LOCAL_TOKENIZER_SHIM=1

Never modifies the installed AIPerf package. Python 'spawn' children inherit
this PYTHONPATH, so they apply the same patch before importing AIPerf services.
"""
import os
from pathlib import Path

if os.environ.get('CJYU_AIPERF_LOCAL_TOKENIZER_SHIM') == '1':
    # Import from installed aiperf; only this specific classmethod is modified.
    # Fail loudly if the target API changed in another aiperf version.
    from aiperf.common.tokenizer import Tokenizer
    if not hasattr(Tokenizer, '_resolve_local_snapshot'):
        raise RuntimeError('Unsupported AIPerf tokenizer API for local-path shim')
    _original = Tokenizer._resolve_local_snapshot.__func__

    @classmethod
    def _resolve_local_snapshot_compat(cls, name, revision):
        if isinstance(name, str):
            p = Path(name).expanduser()
            if p.is_absolute() and p.is_dir():
                return str(p.resolve())
        return _original(cls, name, revision)

    Tokenizer._resolve_local_snapshot = _resolve_local_snapshot_compat
