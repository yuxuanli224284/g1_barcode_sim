"""Process hygiene for Isaac Sim processes.

Problem this solves: upstream unitree_sim_isaaclab imports `logging_mp`, whose
`get_logger()` starts a `multiprocessing.Process` with the default Linux start method `fork`. Forked after the Isaac app
is up, that child is a copy of the whole sim process and inherits its GPU device handles (~4.8 GB per process). It is
only stopped by atexit hooks, which `os._exit()` (used by our scripts because Kit's normal shutdown can hang), a crash
or a SIGKILL all skip. The child then lives on, re-parented to init, holding the GPU memory; nvidia-smi keeps charging it
to the dead parent PID, so it is invisible to `ps -p <pid>`.

`die_with_parent_on_fork()` makes every child forked from this process's main thread get SIGKILL when the parent dies
(Linux PR_SET_PDEATHSIG), whatever way the parent exits. Call it before the Isaac app is launched.
"""
from __future__ import annotations

import ctypes
import os
import signal
import threading

_PR_SET_PDEATHSIG = 1          # <linux/prctl.h>
_installed = False


def die_with_parent_on_fork() -> None:
    """PR_SET_PDEATHSIG fires when the parent *thread* that forked exits, so it is only set for forks made from the main
    thread (whose exit is the process exit); forks from other threads are left alone."""
    global _installed
    if _installed:
        return
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    state = {"main": False}

    def before():
        state["main"] = threading.current_thread() is threading.main_thread()

    def after_in_child():
        if state["main"] and libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
            e = ctypes.get_errno()
            raise OSError(e, f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(e)}")

    os.register_at_fork(before=before, after_in_child=after_in_child)
    _installed = True
