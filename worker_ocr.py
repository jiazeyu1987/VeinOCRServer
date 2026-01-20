import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import socket
import threading
import time
import sys
from datetime import datetime


def init_logger(dst: str = "ocrlog") -> logging.Logger:
    if not os.path.exists(dst):
        os.makedirs(dst)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(dst, f"ocrapp_{today}.log")
    logger = logging.getLogger("worker_ocr")
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


def _import_ocr_detect_source():
    """
    Force-load `ocr_detect.py` (source) even if a compiled extension
    `ocr_detect*.pyd` exists in the same directory.
    """
    import importlib.util
    import os as _os
    import sys as _sys

    existing = _sys.modules.get("ocr_detect")
    if existing is not None and str(getattr(existing, "__file__", "")).endswith("ocr_detect.py"):
        return existing

    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ocr_detect.py")
    spec = importlib.util.spec_from_file_location("ocr_detect", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    _sys.modules["ocr_detect"] = mod
    return mod


def _get_first_present(mapping: dict, keys: list[str], default=0):
    for k in keys:
        if k in mapping and mapping.get(k) is not None:
            return mapping.get(k)
    return default


def _get_online_results(ocrserver) -> dict:
    m = getattr(ocrserver, "MEASSURE", {}) or {}
    depth = _get_first_present(m, ["深度", "娣卞害", "Depth"], 0) or 0
    return {
        "SkinDepth": _get_first_present(m, ["skin_distance"], 0) or 0,
        "A": _get_first_present(m, ["A"], 0) or 0,
        "B": _get_first_present(m, ["B"], 0) or 0,
        "Alpha": _get_first_present(m, ["Alpha"], 0) or 0,
        "Depth": depth,
        "IsFreeze": bool(_get_first_present(m, ["Is_Freeze"], False)),
        "Points_Per_MM": _get_first_present(m, ["Points_Per_MM"], 0) or 0,
    }


class OCRWorker:
    def __init__(self, *, host: str, port: int):
        self.host = host
        self.port = port
        self.logger = init_logger()
        self.setting = load_setting() or {}

        OCRDetect = _import_ocr_detect_source().OCRDetect
        self.ocrserver = OCRDetect(self.setting, self.logger)

        self._stop_event = threading.Event()
        self._last_error: str | None = None
        self._last_error_log_ts = 0.0
        self._last_ok_ts: float | None = None
        self._ocr_thread = threading.Thread(target=self._ocr_loop, daemon=True)
        self._ocr_thread.start()

    def _ocr_loop(self) -> None:
        # Run OCR here and log errors (OCRDetect.start_ocr_server swallows exceptions).
        time_skip = float(getattr(self.ocrserver, "time_skip", 0) or 0)
        if time_skip < 0:
            time_skip = 0

        while not self._stop_event.is_set():
            try:
                self.ocrserver.ocr_instant()
                self._last_ok_ts = time.time()
                if time_skip:
                    self._stop_event.wait(time_skip)
            except Exception as e:
                self._last_error = str(e)
                now = time.time()
                if now - self._last_error_log_ts > 5.0:
                    self._last_error_log_ts = now
                    self.logger.error(f"ocr_instant failed: {e}")
                self._stop_event.wait(0.2)

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
        elif req_type == "ONLINE":
            resp = {
                "success": True,
                "data": _get_online_results(self.ocrserver),
                "meta": {
                    "engine": getattr(self.ocrserver, "ocr_engine", None),
                    "last_ok_s": self._last_ok_ts,
                    "last_error": self._last_error,
                },
            }
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
            self.logger.info(f"OCR worker listening on {self.host}:{self.port}")
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
    p.add_argument("--port", type=int, default=30416)
    args = p.parse_args()

    OCRWorker(host=args.host, port=args.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
