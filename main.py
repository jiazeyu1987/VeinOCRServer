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

# IMPORTANT:
# This repo ships optional Cython extensions (e.g. `server.cp310-...pyd`).
# Python will prefer `.pyd` over `.py` for `import server`, which would bypass
# local edits in `server.py`. Load `server.py` explicitly to ensure the Python
# implementation is used.
_server_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "server.py")
_spec = importlib.util.spec_from_file_location("server_py", _server_path)
server = importlib.util.module_from_spec(_spec)  # type: ignore
assert _spec and _spec.loader
_spec.loader.exec_module(server)  # type: ignore

# server
import socket
import threading
import json
from ocr_detect import OCRDetect
import os
import logging

# ocr
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import pyautogui

import cv2
import time, os


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
