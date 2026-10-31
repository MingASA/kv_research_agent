"""Identity-bound Linux pidfd signalling; never use a PID wildcard/group kill."""

from __future__ import annotations

import os
from pathlib import Path
import signal


# Linux 5.10 include/linux/sched.h; do_exit sets this before cgroup_exit.
# It can be visible while /proc/stat still reports R rather than Z.
LINUX_PF_EXITING = 0x00000004


def process_identity(pid: int) -> dict:
    root = Path("/proc") / str(pid)
    stat_text = (root / "stat").read_text()
    fields = stat_text[stat_text.rfind(")") + 2:].split()
    return {"pid": pid, "uid": root.stat().st_uid, "start_ticks": int(fields[19]),
            "cgroup": (root / "cgroup").read_text(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def process_lifecycle(pid: int) -> dict:
    """Read exit state separately; R/S changes are not process identity changes."""
    root = Path('/proc')/str(pid)
    text = (root/'stat').read_text()
    fields = text[text.rfind(')')+2:].split()
    return {'pid': pid, 'uid': root.stat().st_uid, 'start_ticks': int(fields[19]),
            'state': fields[0], 'kernel_flags': int(fields[6])}


def signal_exact(expected: dict, number: int) -> bool:
    """Signal the captured process instance, returning false if it has exited."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("pidfd identity-safe signalling is required")
    pid = expected["pid"]
    if type(pid) is not int or pid <= 1 or expected.get("uid") != os.getuid():
        raise ValueError("Refusing unowned/invalid process identity")
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        try:
            current = process_identity(pid)
        except FileNotFoundError:
            return False
        if current != expected:
            raise ValueError("PID was reused or process ownership/cgroup changed; no signal sent")
        try:
            signal.pidfd_send_signal(fd, number)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


def pidfd_available() -> bool:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        return False
    try:
        fd = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False
