import os
import subprocess
import sys


_tee = None

def redir_output(logfile: str):
    global _tee
    if _tee:
        raise Exception("Standard output/error already redirected")
    _tee = subprocess.Popen(["tee", logfile], stdin=subprocess.PIPE)
    # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
    # of any child processes we spawn)
    os.dup2(_tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(_tee.stdin.fileno(), sys.stderr.fileno())


def which_python():
    """returns the full path to the python interpreter,
    e.g. /opt/conda/bin/python"""
    which = subprocess.Popen(["which", "python"], stdout=subprocess.PIPE)
    if which.wait():
        raise RuntimeError("which")
    return which.stdout.read().decode()


def get_physical_core_count() -> int:
    """Obtain the number of physical cores on the local machine.
    Logical processors (i.e. hyper-threading cores) will only be counted as ONE processor.
    For example, for a machine with 2 hyperthreads per CPU, and `nproc` reports 32,
    this routine will return 16.

    The return value is suitable for starting numerical-intensive MPI tasks, to avoid
    over-subscribing the float processing units (FPUs), which are usually shared among
    the logical cores.
    """

    lscpu = subprocess.Popen("lscpu", stdout=subprocess.PIPE)
    if lscpu.wait():
        raise RuntimeError("lscpu")
    output = lscpu.stdout.read().decode().splitlines()
    n_cores_per_socket = next(x for x in output if "Core(s) per socket:" in x).split(':')[1].strip()
    n_sockets = next(x for x in output if "Socket(s):" in x).split(':')[1].strip()
    return int(n_cores_per_socket) * int(n_sockets)
