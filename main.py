import os

# Work around OpenMP runtime conflicts on Windows (common with MKL + Paddle/OpenCV).
# Must be set before importing libraries that load OpenMP (e.g., paddlepaddle/paddleocr/numpy).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Force CPU mode for the server entrypoint.
# Must be set before importing Paddle/PaddleOCR.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FLAGS_use_cuda", "0")

import importlib.util
import os as _os
import sys

# IMPORTANT:
# This repo ships optional Cython extensions (e.g. `server.cp310-...pyd`).
# Python will prefer `.pyd` over `.py` for `import server`, which would bypass
# local edits in `server.py`. Load `server.py` explicitly to ensure the Python
# implementation is used.
_base_dir = _os.path.dirname(_os.path.abspath(__file__))

_ocr_detect_path = _os.path.join(_base_dir, "ocr_detect.py")
_ocr_spec = importlib.util.spec_from_file_location("ocr_detect", _ocr_detect_path)
ocr_detect = importlib.util.module_from_spec(_ocr_spec)  # type: ignore
assert _ocr_spec and _ocr_spec.loader
_ocr_spec.loader.exec_module(ocr_detect)  # type: ignore
sys.modules["ocr_detect"] = ocr_detect

_server_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "server.py")
_spec = importlib.util.spec_from_file_location("server", _server_path)
server = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(server)  # type: ignore
sys.modules["server"] = server


if __name__ == '__main__':
    # Override settings.json GPU flag to ensure PaddleOCR runs on CPU.
    _orig_load_setting = server.ImageProcessServer.load_setting

    def _load_setting_cpu(self):
        setting = _orig_load_setting(self)
        if setting is None:
            setting = {}
        setting["GPU"] = False
        return setting

    server.ImageProcessServer.load_setting = _load_setting_cpu
    server.run()
