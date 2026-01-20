import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime

from winjob import JobObject


def init_logger(dst: str = "ocrlog") -> logging.Logger:
    if not os.path.exists(dst):
        os.makedirs(dst)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(dst, f"ocrapp_{today}.log")
    logger = logging.getLogger("server_supervisor")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger


def _setting_candidate_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        if exe_dir:
            dirs.append(exe_dir)
            dirs.append(os.path.dirname(exe_dir))
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here:
            dirs.append(here)
            dirs.append(os.path.dirname(here))
    except Exception:
        pass
    try:
        cwd = os.path.abspath(os.getcwd())
        if cwd:
            dirs.append(cwd)
    except Exception:
        pass

    # de-dup while keeping order
    seen = set()
    out: list[str] = []
    for d in dirs:
        if not d:
            continue
        d = os.path.normpath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def load_setting() -> dict | None:
    for d in _setting_candidate_dirs():
        setting_path = os.path.join(d, "settings")
        if not os.path.exists(setting_path):
            continue
        for enc in ("utf-8", "utf-8-sig"):
            try:
                with open(setting_path, "r", encoding=enc) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _send_worker(port: int, payload: dict, timeout_s: float = 2.0) -> dict | None:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as s:
            s.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            s.settimeout(timeout_s)
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    data = data.split(b"\n", 1)[0]
                    break
            if not data:
                return None
            return json.loads(data.decode("utf-8"))
    except Exception:
        return None


class WorkerSpec:
    def __init__(self, *, name: str, script: str, port: int, enabled: bool = True):
        self.name = name
        self.script = script
        self.port = port
        self.enabled = enabled


class SupervisorServer:
    def __init__(self, *, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = init_logger()
        self.setting = load_setting() or {}

        self._stop_event = threading.Event()
        self._job = JobObject()
        self._workers_lock = threading.Lock()
        self._workers: dict[str, subprocess.Popen] = {}
        self._restart_backoff_s: dict[str, float] = {}

        peak_cfg = self.setting.get("peak_detection", {}) if isinstance(self.setting, dict) else {}
        self.ocr_worker_port = int(peak_cfg.get("ocr_worker_port", 30416))
        self.compare_worker_port = int(peak_cfg.get("compare_worker_port", 30417))

        ocr_enabled = True
        if isinstance(self.setting, dict):
            ocr_engine = str(self.setting.get("ocr", {}).get("engine", "")).lower()
            if ocr_engine in ("none", "off", "disabled"):
                ocr_enabled = False

        self.worker_specs = [
            WorkerSpec(name="ocr", script="worker_ocr.py", port=self.ocr_worker_port, enabled=ocr_enabled),
            WorkerSpec(name="compare", script="worker_compare.py", port=self.compare_worker_port, enabled=True),
        ]

    def _spawn_worker(self, spec: WorkerSpec) -> None:
        if not spec.enabled:
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if getattr(sys, "frozen", False):
            # In PyInstaller builds, sys.executable is the bundled EXE.
            # Re-invoking it with a .py path will just start the supervisor again.
            # Use an explicit role flag so main.py can dispatch to worker code.
            cmd = [
                sys.executable,
                "--role",
                spec.name,
                "--host",
                "127.0.0.1",
                "--port",
                str(spec.port),
            ]
        else:
            script_path = os.path.join(base_dir, spec.script)
            cmd = [
                sys.executable,
                script_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(spec.port),
            ]

        self.logger.info(f"Starting worker {spec.name}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=base_dir)

        try:
            # On Windows, Popen exposes the native process handle via `_handle`.
            self._job.assign_process_handle(proc._handle)  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.error(f"Assign worker {spec.name} to JobObject failed: {e}")

        with self._workers_lock:
            self._workers[spec.name] = proc
            self._restart_backoff_s.setdefault(spec.name, 1.0)

    def _ensure_workers_started(self) -> None:
        for spec in self.worker_specs:
            if not spec.enabled:
                continue
            with self._workers_lock:
                proc = self._workers.get(spec.name)
            if proc is not None and proc.poll() is None:
                continue
            self._spawn_worker(spec)

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            for spec in self.worker_specs:
                if not spec.enabled:
                    continue
                with self._workers_lock:
                    proc = self._workers.get(spec.name)
                if proc is None:
                    continue
                rc = proc.poll()
                if rc is None:
                    continue

                backoff = self._restart_backoff_s.get(spec.name, 1.0)
                self.logger.error(f"Worker {spec.name} exited (code={rc}); restart in {backoff:.1f}s")
                time.sleep(backoff)
                self._restart_backoff_s[spec.name] = min(backoff * 2.0, 30.0)
                if self._stop_event.is_set():
                    break
                self._spawn_worker(spec)

            time.sleep(0.2)

    def _shutdown_workers(self) -> None:
        with self._workers_lock:
            procs = list(self._workers.items())
            self._workers.clear()

        for name, proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
            self.logger.info(f"Terminated worker {name}")

        try:
            self._job.close()
        except Exception:
            pass

    def _handle_client(self, client_socket: socket.socket, client_address) -> None:
        self.logger.info(f"client connected: {client_address}")
        try:
            while not self._stop_event.is_set():
                request = client_socket.recv(1024).decode("utf-8").strip()
                if not request:
                    break

                parts = request.split(";")
                req_type = parts[0].upper()
                password = parts[1] if len(parts) > 1 else None
                arg = parts[2] if len(parts) > 2 else None

                if password != "31415":
                    response = "密码错误"
                else:
                    if req_type == "ONLINE":
                        if any(s.name == "ocr" and s.enabled for s in self.worker_specs):
                            resp = _send_worker(self.ocr_worker_port, {"type": "ONLINE"})
                            if isinstance(resp, dict) and isinstance(resp.get("data"), dict):
                                response = dict(resp["data"])
                                if isinstance(resp.get("meta"), dict):
                                    response["_meta"] = resp["meta"]
                            else:
                                response = resp or {"success": False, "error": "ocr worker no response"}
                        else:
                            response = {
                                "SkinDepth": 0,
                                "A": 0,
                                "B": 0,
                                "Alpha": 0,
                                "Depth": 0,
                                "IsFreeze": False,
                                "Points_Per_MM": 0,
                            }
                    elif req_type == "OFFLINE":
                        try:
                            payload = json.loads(arg) if arg else {}
                        except Exception:
                            payload = {}
                        point_id = payload.get("point_id")
                        is_save = bool(payload.get("is_save", True))
                        _send_worker(self.compare_worker_port, {"type": "OFFLINE", "point_id": point_id, "is_save": is_save})
                        response = None  # keep legacy behavior: OFFLINE doesn't reply
                    elif req_type == "CLOSE":
                        response = {"success": True, "info": "close successfully"}
                        self._stop_event.set()
                    else:
                        response = {
                            "success": False,
                            "info": f"错误: 未知请求类型 '{req_type}'。",
                        }

                if response:
                    raw = json.dumps(response, ensure_ascii=False)
                    client_socket.send(raw.encode("utf-8"))
        except Exception as e:
            self.logger.error(f"handle client {client_address} error: {e}")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    def serve_forever(self) -> None:
        self._ensure_workers_started()
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            self.logger.info(f"Supervisor server listening on {self.host}:{self.port}")

            try:
                while not self._stop_event.is_set():
                    client_socket, client_address = server_socket.accept()
                    t = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address),
                        daemon=True,
                    )
                    t.start()
            except KeyboardInterrupt:
                self.logger.info("Supervisor server interrupted")
            finally:
                self._stop_event.set()
                self._shutdown_workers()


def run(host: str = "127.0.0.1", port: int = 30415) -> None:
    SupervisorServer(host=host, port=port).serve_forever()


if __name__ == "__main__":
    run("127.0.0.1", 30415)
