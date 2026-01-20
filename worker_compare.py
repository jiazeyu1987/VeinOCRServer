import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import socket
import threading
import sys
from datetime import datetime


def init_logger(dst: str = "ocrlog") -> logging.Logger:
    if not os.path.exists(dst):
        os.makedirs(dst)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(dst, f"ocrapp_{today}.log")
    logger = logging.getLogger("worker_compare")
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


def _load_compare_points_class():
    """
    Force-load `treat_compare_img.py` (source) even if a compiled extension
    `treat_compare_img*.pyd` exists in the same directory.
    """
    import importlib.util
    import os as _os

    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "treat_compare_img.py")
    spec = importlib.util.spec_from_file_location("treat_compare_img_py", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod.ComparePoints


class CompareWorker:
    def __init__(self, *, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = init_logger()
        self.setting = load_setting() or {}

        ComparePoints = _load_compare_points_class()
        self.compareTool = ComparePoints(self.setting, self.logger)

        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self.compareTool.monitor_peaks,
            args=(self._stop_event,),
            daemon=True,
        )
        self._monitor_thread.start()

    def _handle_one(self, conn: socket.socket) -> None:
        raw = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            raw += chunk
            if b"\n" in raw:
                raw = raw.split(b"\n", 1)[0]
                break

        if not raw:
            return

        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            return

        req_type = str(req.get("type", "")).upper()
        if req_type == "PING":
            resp = {"success": True, "type": "PONG"}
        elif req_type == "OFFLINE":
            point_id = req.get("point_id")
            is_save = bool(req.get("is_save", True))
            try:
                resp = self.compareTool.save_latest(point_id=point_id, is_save=is_save)
            except Exception as e:
                resp = {"success": False, "error": str(e)}
        elif req_type == "SHUTDOWN":
            self._stop_event.set()
            resp = {"success": True, "type": "SHUTDOWN"}
        else:
            resp = {"success": False, "error": f"Unknown type: {req_type}"}

        try:
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            pass

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(16)
            self.logger.info(f"Compare worker listening on {self.host}:{self.port}")
            s.settimeout(0.5)

            while not self._stop_event.is_set():
                try:
                    conn, _addr = s.accept()
                except socket.timeout:
                    continue
                except Exception:
                    continue
                with conn:
                    self._handle_one(conn)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30417)
    args = p.parse_args()

    CompareWorker(host=args.host, port=args.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
