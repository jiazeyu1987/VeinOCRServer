# -*- coding: utf-8 -*-
import os
import argparse
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageDraw

import paddle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paddle/PaddleOCR sanity check (CPU/GPU).")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-cpu", action="store_true", help="Run on CPU")
    group.add_argument("-gpu", action="store_true", help="Run on GPU (requires CUDA build)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    use_gpu = bool(args.gpu)

    print("paddle:", paddle.__version__)
    print("is_compiled_with_cuda:", paddle.is_compiled_with_cuda())
    print("device before:", paddle.device.get_device())

    if use_gpu:
        if not paddle.is_compiled_with_cuda():
            print("FAIL: 当前 paddle 不是 CUDA/GPU 版本，无法使用 -gpu。")
            return 2
        paddle.set_device("gpu:0")
    else:
        # Force CPU mode (even if CUDA is available/installed).
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        paddle.set_device("cpu")
    print("device after:", paddle.device.get_device())

    if use_gpu:
        try:
            cc_major, cc_minor = paddle.device.cuda.get_device_capability()
            print(f"gpu compute capability: {cc_major}.{cc_minor}")
        except Exception:
            pass
        try:
            print("paddle cuda:", paddle.version.cuda())
        except Exception:
            pass
        try:
            print("paddle cudnn:", paddle.version.cudnn())
        except Exception:
            pass

    # 1) Pure Paddle compute sanity check
    x = paddle.randn([1024, 1024])
    y = paddle.matmul(x, x)
    if use_gpu:
        paddle.device.cuda.synchronize()
    print("paddle matmul ok, mean:", float(y.mean()))

    # 2) PaddleOCR end-to-end sanity check (will download models on first run)
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        print("FAIL: import paddleocr failed (often NumPy/OpenCV binary mismatch on Windows).")
        print("Fix: reinstall pinned wheels from `requirement.txt` (NumPy 1.26.4 + OpenCV <4.12 + PaddleOCR <3).")
        print("Error:", repr(e))
        return 3
    except OSError as e:
        print("FAIL: import paddleocr failed (likely pulled in torch DLL deps on Windows).")
        print("Fix: reinstall PaddleOCR 2.x, e.g. `pip install \"paddleocr<3\"` (see requirement.txt).")
        print("Error:", repr(e))
        return 3

    img = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), "测试123 ABC", fill=(0, 0, 0))
    np_img = np.array(img)

    ocr = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=use_gpu, show_log=False)
    t0 = time.time()
    res = ocr.ocr(np_img, cls=True)
    if use_gpu:
        paddle.device.cuda.synchronize()
    print("paddleocr ok, elapsed(s):", round(time.time() - t0, 3))
    print("result sample:", (res[0][:2] if res and res[0] else res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
