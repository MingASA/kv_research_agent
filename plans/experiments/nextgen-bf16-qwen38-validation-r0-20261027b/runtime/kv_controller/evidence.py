"""Local imports for tests; deployment freezes audit_native_results.py verbatim."""

from scripts.audit_native_results import (
    _close, _markers, _percentile, _value, read_records, telemetry_output,
    verify_allocation, verify_backend, verify_summary,
)
