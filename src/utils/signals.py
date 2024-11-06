import datetime
import os
import re
import signal
import sys
import threading
import traceback
import types
# from concurrent.futures import ThreadPoolExecutor
from io import StringIO
# from multiprocessing import Lock, Process
from os import getpid
# from time import sleep
from typing import Callable, Optional

import psutil

# LOCK = Lock()


def _signal_tree(pid: int, sig: signal.Signals) -> None:
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    print("====  Found {} child processes for PID {}: {} ====".format(len(children), pid, children))
    for p in children:
        try:
            p.send_signal(sig)
        except psutil.NoSuchProcess:
            pass


class TeeWriter:
    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._stream = sys.stdout

    def __enter__(self) -> "TeeWriter":
        self._file = open(self._filename, "a")
        return self

    def write(self, data: str) -> None:
        self._file.write(data)
        self._stream.write(data)

    def flush(self) -> None:
        self._file.flush()
        self._stream.flush()

    def __exit__(self, *args, **kwargs):  # type: ignore
        self._file.close()


def _create_handler(
    pass_through: bool, match: Optional[str] = None
) -> Callable[[int, Optional[types.FrameType]], None]:
    """
    if pass_through, pass the signal to all child processes as well
    """

    filename = f"stacktraces-{getpid()}.log"

    if "AMLT_OUTPUT_DIR" in os.environ:
        filename = os.path.join(os.environ["AMLT_OUTPUT_DIR"], filename)

    def handler(signal_number: int, frame: Optional[types.FrameType]) -> None:
        with TeeWriter(filename) as file:
            print(
                f"---- Process {getpid()} received signal {signal_number}: {datetime.datetime.now().isoformat()} ----",
                file=file,
            )
            threads = list(threading.enumerate())
            tracebacks = {}
            for th in threads:
                buf = StringIO()
                assert th.ident
                traceback.print_stack(sys._current_frames()[th.ident], file=buf)
                tracebacks[str(th)] = buf.getvalue()
            print(f"------ Found {len(tracebacks)} threads in process {getpid()}.", file=file)
            for info, tb in tracebacks.items():
                if match and not re.search(match, tb):
                    print(f"Skipping {info} because it does not match {match}", file=file)
                    continue
                print("---------", info, file=file)
                print(tb, file=file)

            if pass_through:
                print("Signaling children of process", getpid(), file=file)
                _signal_tree(getpid(), signal.SIGUSR2)

    return handler


def register_print_stack_on_sigusr2(pass_through: bool = True, match: Optional[str] = None) -> None:
    """
    To investigate hanging processes, call this function once in your main process
    with pass_through=True, and in every subprocess with pass_through=False.

    If `match` is specified, only print stack traces matching this pattern.
    """
    for sig in [signal.SIGUSR2]:
        signal.signal(sig, _create_handler(pass_through))

