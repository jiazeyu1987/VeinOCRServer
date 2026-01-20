import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
# Must be set before importing libraries that load OpenMP (e.g., paddlepaddle/paddleocr/numpy).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Force CPU mode for the server entrypoint.
# Must be set before importing Paddle/PaddleOCR.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FLAGS_use_cuda", "0")

import argparse
import importlib.util
import os as _os
import sys
import subprocess
import time

# IMPORTANT:
# This repo ships optional Cython extensions (e.g. `server.cp310-...pyd`).
# Python will prefer `.pyd` over `.py` for `import server`, which would bypass
# local edits in `server.py`. Load `server.py` explicitly to ensure the Python
# implementation is used.
_base_dir = _os.path.dirname(_os.path.abspath(__file__))

def _parse_role(argv: list[str]) -> tuple[str | None, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--role", choices=["ocr", "compare", "supervisor"], default=None)
    ns, rest = p.parse_known_args(argv[1:])
    return ns.role, [argv[0], *rest]


def _load_server_py():
    _server_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "server.py")
    _spec = importlib.util.spec_from_file_location("server", _server_path)
    mod = importlib.util.module_from_spec(_spec)  # type: ignore
    assert _spec and _spec.loader
    _spec.loader.exec_module(mod)  # type: ignore
    sys.modules["server"] = mod
    return mod


def _pids_listening_on_port(port: int) -> set[int]:
    try:
        out = subprocess.check_output(["cmd", "/c", "netstat -ano -p tcp"], text=True, errors="ignore")
    except Exception:
        return set()

    pids: set[int] = set()
    needle = f":{port} "
    for line in out.splitlines():
        if needle not in line:
            continue
        if "LISTENING" not in line.upper():
            continue
        parts = line.split()
        if not parts:
            continue
        try:
            pid = int(parts[-1])
        except Exception:
            continue
        if pid > 0:
            pids.add(pid)
    return pids


def _get_process_cmdline(pid: int) -> str:
    cmd = (
        f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\";"
        "if ($p -and $p.CommandLine) { $p.CommandLine }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            errors="ignore",
        )
        return (out or "").strip()
    except Exception:
        return ""


def _is_veinocr_cmd(cmdline: str) -> bool:
    if not cmdline:
        return False
    c = cmdline.lower()
    if "veinocrserver" in c:
        return True
    for name in ("main.py", "server.py", "server_supervisor.py", "worker_ocr.py", "worker_compare.py"):
        if name in c:
            return True
    return False


def _kill_pid_tree(pid: int) -> None:
    try:
        subprocess.check_call(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _cleanup_previous_instances() -> None:
    # Avoid process explosion by killing older instances occupying our ports.
    ports = [30415, 30416, 30417]
    killed = 0
    for port in ports:
        for pid in sorted(_pids_listening_on_port(port)):
            if pid == os.getpid():
                continue
            cmdline = _get_process_cmdline(pid)
            if not _is_veinocr_cmd(cmdline):
                print(f"[main] Port {port} used by PID {pid}, not VeinOCRServer; skip.")
                continue
            print(f"[main] Killing previous PID {pid} (port {port})")
            _kill_pid_tree(pid)
            killed += 1
    if killed:
        time.sleep(0.5)


if __name__ == '__main__':
    role, new_argv = _parse_role(sys.argv)
    sys.argv = new_argv

    if role == "ocr":
        import worker_ocr

        raise SystemExit(worker_ocr.main())
    if role == "compare":
        import worker_compare

        raise SystemExit(worker_compare.main())
    if role == "supervisor":
        import server_supervisor

        server_supervisor.run("127.0.0.1", 30415)
        raise SystemExit(0)

    # Default: supervisor entrypoint (via server.run()).
    _cleanup_previous_instances()
    server = _load_server_py()
    server.run()
