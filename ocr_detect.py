import numpy as np
import pyautogui

import cv2
import time, os
import re
from typing import Dict, List, Optional


def _load_paddleocr():
    from paddleocr import PaddleOCR  # type: ignore

    draw_ocr = None
    try:
        from paddleocr import draw_ocr as _draw_ocr  # type: ignore

        draw_ocr = _draw_ocr
    except Exception:
        draw_ocr = None

    return PaddleOCR, draw_ocr


def _load_pytesseract():
    import pytesseract  # type: ignore

    return pytesseract


def _pil_from_rgb(rgb: np.ndarray):
    from PIL import Image  # type: ignore

    return Image.fromarray(rgb)


class OCRDetect:
    def __init__(self, setting=None, logger=None):

        self.logger = logger
        if logger is None:
            import logging
            self.logger = logging.getLogger('OCRDetect')

        if setting is None:
            setting = {'GPU': True, "time_skip": 0}

        cur_dir = os.path.dirname(os.path.abspath(__file__))

        ocr_cfg = setting.get("ocr", {}) if isinstance(setting, dict) else {}
        self.ocr_engine = str(ocr_cfg.get("engine", "tesseract")).lower().strip()
        self.fallback_to_paddle = bool(ocr_cfg.get("fallback_to_paddle", True))
        self.tesseract_cfg = ocr_cfg.get("tesseract", {}) if isinstance(ocr_cfg, dict) else {}

        use_gpu = setting['GPU'] if 'GPU' in setting else False
        if use_gpu is True:
            self.logger.info("Using GPU checking.")
            gpu_num = self.get_gpu_count()
            if gpu_num == 0:
                use_gpu = False
        self._use_gpu = use_gpu

        self.OCR_MDOEL = None
        self._draw_ocr = None
        self._pytesseract = None

        self._paddle_kwargs = dict(
            use_angle_cls=True,
            lang="ch",
            use_gpu=use_gpu,
            rec_image_shape='3, 24, 160',
            rec_batch_num=8,
            precision='fp32',
            det_model_dir=setting['det'] if 'det' in setting else './whl/det/ch/ch_PP-OCRv4_det_infer',
            rec_model_dir=setting['rec'] if 'rec' in setting else './whl/rec/ch/ch_PP-OCRv4_rec_infer',
            cls_model_dir=setting['cls'] if 'cls' in setting else './whl/cls/ch_ppocr_mobile_v2.0_cls_infer',
        )
        if 'log' in setting:
            self._paddle_kwargs["show_log"] = setting['log']

        if self.ocr_engine in ("none", "off", "disabled"):
            self.logger.info("OCR engine: disabled")
            self.ocr_engine = "none"
        elif self.ocr_engine == "tesseract":
            try:
                self._pytesseract = _load_pytesseract()
                tcmd = str(self.tesseract_cfg.get("tesseract_cmd", "")).strip()
                if tcmd:
                    self._pytesseract.pytesseract.tesseract_cmd = tcmd
                self.logger.info("OCR engine: tesseract")
            except Exception as e:
                self.logger.error(f"tesseract init failed: {e}")
                if not self.fallback_to_paddle:
                    raise
                self.ocr_engine = "paddle"

        if self.ocr_engine == "paddle":
            self._init_paddleocr()

            print(os.path.join(cur_dir, 'whl/cls/ch_ppocr_mobile_v2.0_cls_infer'))
            self.logger.info("OCR engine: paddle")

        self.time_skip = setting['time_skip'] if 'time_skip' in setting else 0

        # 测量，缩放，是否冻结等尺度相关
        self.MEASSURE = {'增益': None, '深度': None, '频率': None, '图像增强': None,

                    'skin_distance': None, 'A': None, 'B': None, 'Alpha': None, 'Zoom_scaler': 1.0, 'Is_Freeze': False}

    def _init_paddleocr(self):
        if self.OCR_MDOEL is not None:
            return
        PaddleOCR, draw_ocr = _load_paddleocr()
        self._draw_ocr = draw_ocr
        # PaddleOCR 2.x uses `use_gpu=...`; PaddleOCR 3.x removed/renamed several args and
        # validates "common args", raising `ValueError: Unknown argument: xxx`.
        # Keep a single codepath by (1) setting Paddle device explicitly, (2) retrying with
        # unsupported kwargs removed.
        os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

        try:
            import paddle  # type: ignore

            if self._use_gpu and getattr(paddle, "is_compiled_with_cuda", lambda: False)():
                paddle.set_device("gpu:0")
            else:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                paddle.set_device("cpu")
        except Exception:
            pass

        paddle_kwargs = dict(self._paddle_kwargs)
        # Some envs/configs pass `show_log`; keep it best-effort.
        max_attempts = 20
        for _ in range(max_attempts):
            try:
                self.OCR_MDOEL = PaddleOCR(**paddle_kwargs)
                return
            except Exception as e:
                msg = str(e)
                # PaddleOCR 3.x: ValueError("Unknown argument: use_gpu")
                m = re.search(r"Unknown argument:\s*([A-Za-z_][A-Za-z0-9_]*)", msg)
                if m:
                    bad = m.group(1)
                    if bad in paddle_kwargs:
                        paddle_kwargs.pop(bad, None)
                        continue
                # Python TypeError: got an unexpected keyword argument 'xxx'
                m2 = re.search(r"unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", msg)
                if m2:
                    bad = m2.group(1)
                    if bad in paddle_kwargs:
                        paddle_kwargs.pop(bad, None)
                        continue
                # Last-resort: some builds don't support show_log.
                if "show_log" in paddle_kwargs:
                    paddle_kwargs.pop("show_log", None)
                    continue
                raise
        raise RuntimeError(f"Failed to init PaddleOCR after {max_attempts} attempts; last kwargs={sorted(paddle_kwargs.keys())}")


    def get_gpu_count(self):
        import pynvml
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            print(f"可用的 GPU 数量: {count}")

            # 打印每个 GPU 的信息
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetName(handle)
                print(f"GPU {i}: {info}")
                self.logger.info(f"GPU {i}: {info}")

            pynvml.nvmlShutdown()
            return count
        except pynvml.NVMLError as e:
            print(f"NVML 错误: {e}")
            return 0


    def showimg(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(0)
    def results_show(self, img, result):
        # 显示结果
        result = result[0]
        boxes = []
        txts = []
        scores = []
        if result is not None:
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
        if self._draw_ocr is None:
            return
        im_show = self._draw_ocr(img, boxes, txts, scores, font_path='./fonts/simfang.ttf')
        cv2.imshow('img1', im_show)
        cv2.waitKey(0)

    def _preprocess_for_tesseract(self, rgb: np.ndarray):
        cfg = self.tesseract_cfg.get("preprocess", {}) if isinstance(self.tesseract_cfg, dict) else {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return rgb

        img = rgb
        try:
            upscale = int(cfg.get("upscale", 2))
            if upscale > 1:
                img = cv2.resize(img, (img.shape[1] * upscale, img.shape[0] * upscale), interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            th_mode = str(cfg.get("threshold", "otsu")).lower().strip()
            if th_mode == "otsu":
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif th_mode == "binary":
                _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            else:
                bw = gray

            bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
            return bw
        except Exception:
            return rgb

    def _tesseract_ocr(self, bgr: np.ndarray, *, psm: Optional[int] = None):
        if self._pytesseract is None:
            raise RuntimeError("pytesseract is not available")

        if bgr is None:
            return [[]]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if len(bgr.shape) == 3 else bgr
        rgb = self._preprocess_for_tesseract(rgb)
        pil_img = _pil_from_rgb(rgb)

        lang = str(self.tesseract_cfg.get("lang", "chi_sim")).strip() or "chi_sim"
        oem = int(self.tesseract_cfg.get("oem", 3))
        base_psm = int(self.tesseract_cfg.get("psm", 6))
        use_psm = int(psm) if psm is not None else base_psm
        extra = str(self.tesseract_cfg.get("config", "")).strip()
        whitelist = str(self.tesseract_cfg.get("whitelist", "")).strip()

        config_parts = [f"--oem {oem}", f"--psm {use_psm}"]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        if extra:
            config_parts.append(extra)
        config = " ".join(config_parts)

        data: Dict[str, List[str]] = self._pytesseract.image_to_data(
            pil_img,
            lang=lang,
            config=config,
            output_type=self._pytesseract.Output.DICT,
        )

        n = len(data.get("text", []))
        items = []
        for i in range(n):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            score = 0.0 if conf < 0 else max(0.0, min(1.0, conf / 100.0))

            try:
                x = int(float(data["left"][i]))
                y = int(float(data["top"][i]))
                w = int(float(data["width"][i]))
                h = int(float(data["height"][i]))
            except Exception:
                continue

            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            items.append([box, (text, score)])

        return [items]

    def _ocr(self, img_bgr: np.ndarray, *, line_mode: bool = False):
        if img_bgr is None:
            return [[]]

        if self.ocr_engine == "tesseract":
            psm = None
            if line_mode:
                psm = int(self.tesseract_cfg.get("line_psm", 7))
            try:
                return self._tesseract_ocr(img_bgr, psm=psm)
            except Exception as e:
                if not self.fallback_to_paddle:
                    raise
                self.logger.error(f"tesseract ocr failed, fallback to paddle: {e}")
                self.ocr_engine = "paddle"
                self._init_paddleocr()

        if self.OCR_MDOEL is None:
            raise RuntimeError("PaddleOCR model is not initialized")
        return self.OCR_MDOEL.ocr(img_bgr)

    def find_is_freeze_in_ocr_results(self, results):
        ## 超声图像中的位置
        # 是否冻结：冻结的话，有一个雪花的标志，会被识别到一个 * 符号
        freeze_pos_col_lu = 1528 - 1304 # 286；該results的image相對於全尺寸的img的位置
        freeze_pos_row_lu = 800 - 822 # = 46；該results的image相對於全尺寸的img的位置
        freeze_pos_col_rb = 1660 - 1304  # 286；該results的image相對於全尺寸的img的位置
        freeze_pos_row_rb = 938 - 822  # = 46；該results的image相對於全尺寸的img的位置

        freeze_text = ["*", "米", "焦深", "管宽"]

        for batch_id in range(len(results)):
            batch = results[batch_id]
            if batch is None:
                return False

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]

                if text in freeze_text:# and freeze_pos_col_lu <= positions[0][0] and freeze_pos_row_lu <= positions[0][1] and freeze_pos_col_rb >= positions[1][0] and freeze_pos_row_rb >= positions[-1][1]:#and freeze_pos_col >= positions[0][0] and freeze_pos_col <= positions[1][0] and freeze_pos_row >= positions[0][1] and freeze_pos_row <= positions[-1][1]:
                    return False

        return True

    def detect_distance_in_img(self, img):
        ##
        # A_img = img[152:180, 1581:1704]
        # A_img = img[194:217, 1581:1704]

        A_row_start = 152
        A_row_end = 217
        A_col_start = 1555
        A_col_end = 1704

        B_row_start = 152 # 先假定在上面
        B_row_end = 180 # 先假定在上面

        B_col_start = 1743 # fixed; 冒号之后
        B_col_end = 1836 # fixed

        alpha_row_start = B_row_start
        alpha_row_end = B_row_end
        alpha_col_start = 1874
        alpha_col_end = 1907


        A_img = img[A_row_start:A_row_end, A_col_start:]
        try:
            A_img = cv2.resize(A_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            A_img = cv2.convertScaleAbs(A_img, alpha=1.3, beta=0)
        except Exception:
            pass

        # self.showimg(A_img)

        results = self._ocr(A_img, line_mode=True)

        # self.results_show(A_img, results)

        batch = results[0]  #输入的图片就一张，batchsize=1

        # print(batch)
        # 没有识别到
        if batch is None:
            return None

        tokens = []
        for item in batch:
            if item is None:
                continue
            try:
                box = item[0]
                t = item[1][0]
            except Exception:
                continue
            if t is None:
                continue
            t = str(t).strip()
            if not t:
                continue
            try:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x0 = min(xs)
                y0 = min(ys)
            except Exception:
                x0, y0 = 0, 0
            tokens.append((y0, x0, t))

        tokens.sort(key=lambda it: (it[0], it[1]))
        text_line = "".join([t for _, _, t in tokens])
        parsed = self._parse_distance_overlay(text_line)
        if "A" in parsed and parsed["A"] is not None:
            self.MEASSURE["A"] = parsed["A"]
        if "B" in parsed and parsed["B"] is not None:
            self.MEASSURE["B"] = parsed["B"]
        if "skin_distance" in parsed and parsed["skin_distance"] is not None:
            self.MEASSURE["skin_distance"] = parsed["skin_distance"]
        if "Alpha" in parsed and parsed["Alpha"] is not None:
            self.MEASSURE["Alpha"] = parsed["Alpha"]


        # results_show(A_img, results)

    def _parse_mm_value(self, s: str):
        if s is None:
            return None
        raw = str(s).strip()
        if not raw:
            return None

        raw = (
            raw.replace("O", "0")
            .replace("o", "0")
            .replace(",", ".")
            .replace("I", "1")
            .replace("l", "1")
            .replace("|", "1")
            .replace("!", "1")
        )
        raw = raw.replace(" ", "")
        allowed = "0123456789.-"
        filtered = "".join(ch for ch in raw if ch in allowed)
        if not filtered:
            return None

        try:
            if "." not in filtered and filtered.lstrip("-").isdigit():
                digits = filtered.lstrip("-")
                sign = "-" if filtered.startswith("-") else ""
                if len(digits) >= 3:
                    filtered = f"{sign}{digits[:-2]}.{digits[-2:]}"
            return float(filtered)
        except Exception:
            return None

    def _parse_number_value(self, s: str):
        if s is None:
            return None
        raw = str(s).strip()
        if not raw:
            return None
        raw = (
            raw.replace("O", "0")
            .replace("o", "0")
            .replace(",", ".")
            .replace("I", "1")
            .replace("l", "1")
            .replace("|", "1")
            .replace("!", "1")
            .replace(" ", "")
        )
        allowed = "0123456789.-"
        filtered = "".join(ch for ch in raw if ch in allowed)
        if not filtered:
            return None
        try:
            return float(filtered)
        except Exception:
            return None

    def _parse_distance_overlay(self, text: str):
        out = {}
        if text is None:
            return out

        s = str(text)
        s = s.replace("：", ":").replace(" ", "")
        # Join split digit tokens like "0.0" + "2mm" -> "0.02mm"
        s = re.sub(r"(?<=\\d)[,，](?=\\d)", "", s)
        s = re.sub(r"(?<=\\d)\\.(?:,|，)(?=\\d)", ".", s)
        # Normalize a few common OCR confusions
        s = s.replace("O", "0").replace("o", "0")

        m = re.search(r"距离:([0-9Oo.,-]+)", s)
        if m:
            out["skin_distance"] = self._parse_mm_value(m.group(1))

        m = re.search(r"A:([0-9Oo.,-]+)", s)
        if m:
            out["A"] = self._parse_mm_value(m.group(1))

        m = re.search(r"B:([0-9Oo.,-]+)", s)
        if m:
            out["B"] = self._parse_mm_value(m.group(1))

        # alpha may be α or a
        m = re.search(r"(?:α|a|c|o):([0-9Oo.,-]+)", s)
        if m:
            out["Alpha"] = self._parse_number_value(m.group(1))
        elif "Alpha" not in out:
            m = re.search(r"([0-9Oo.,-]+)°", s)
            if m:
                out["Alpha"] = self._parse_number_value(m.group(1))

        if ("A" not in out) or ("B" not in out):
            mm_candidates = re.findall(r"([0-9Oo.,-]+)mm", s)
            mm_values = []
            for cand in mm_candidates:
                v = self._parse_mm_value(cand)
                if v is not None:
                    mm_values.append(v)

            # If distance was parsed, try to drop the closest match from candidates.
            dist = out.get("skin_distance")
            if dist is not None and mm_values:
                closest_idx = min(range(len(mm_values)), key=lambda i: abs(mm_values[i] - dist))
                mm_values = [v for i, v in enumerate(mm_values) if i != closest_idx]

            # Use remaining values as A/B fallback (prefer last two in case distance is still present).
            if len(mm_values) >= 2:
                if "A" not in out:
                    out["A"] = mm_values[-2]
                if "B" not in out:
                    out["B"] = mm_values[-1]

        return out


    def find_Zoom_Scaler_in_ocr_results(self, results):
        ## 超声图像中的位置
        # HIFU模式下的测试距离
        # x = (447 + 646) / 2
        # y = (92 + 115) / 2
        target_text = "缩放倍数"

        for batch_id in range(len(results)):
            batch = results[batch_id]

            if batch is None:
                return 1.0

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]

                text = text.replace("：", ":")
                if target_text in text:
                    # if x >= positions[0][0] and x <= positions[1][0] and y >= positions[0][1] and y <= positions[-1][1]:

                    startid = text.index(':') + 1 #英文的冒号

                    return float(text[startid:])
        return None # 没识别到

    def find_text_at_designated_postion_in_ocr_results(self, results, target_positions=[0,0]): # 找到给定position下方的数值

        position = [0, 0]
        position[0] = (target_positions[0][0] + target_positions[1][0]) // 2
        position[1] = target_positions[-1][1] + 20  # 矩形的最后一个点的y，或者第三个都行；加上的数值，需要>=15 and <=20

        for batch_id in range(len(results)):
            batch = results[batch_id]

            if batch is None:
                return None

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]

                # 下面矩形，右侧竖直线 穿过上面的矩形                或者              下面矩形，左侧的数值先，穿过上面的矩形
                if target_positions[0][0] < positions[1][0] < target_positions[1][0] or target_positions[0][0] < positions[0][0] < target_positions[1][0]:
                    if  positions[0][1] <= position[1] <= positions[-1][1]:
                        return text
        return None

    def find_other_setting_in_ocr_results(self, results):
        # 搞一个全局的setting，实时的更新这些数值，可能随时会被用到
        SETTINGS = {'增益': None, '深度': None, '频率': None, '图像增强': None}  # units: dB, cm, MHz        '频率': None, 'MHz': None,

        # {'增益': '60', '深度': '6.3', '频率': '8.3', '图像增强': None}; 图像增强没有识别到，因为下方的数字的框太小了
        # 增益， 深度， 动态范围，频率， 图像增强， 灰阶图谱，
        for batch_id in range(len(results)):
            batch = results[batch_id]
            if batch is None:
                return SETTINGS

            for i in range(len(batch)): # 每张图片中识别的所有list
                positions = batch[i][0]
                text = batch[i][1][0]
                prob = batch[i][1][1]



                for set_text in SETTINGS.keys():
                    if set_text in text: # 说明找到了需要的设置，这些text的下面就是要获取的数值
                        # if '深度' in text:
                        #     print(positions)
                        value = self.find_text_at_designated_postion_in_ocr_results(results, positions)
                        # value may be None
                        SETTINGS[set_text] = value

                        if value is not None:
                            try:
                                value = float(value)
                                SETTINGS[set_text] = value
                            except:
                                SETTINGS[set_text] = None

                            # SETTINGS[set_text] = float(value)
        return SETTINGS
    def cal_points_per_mm(self, deepth, zoom_scaler):
        # 计算单位mm，的像素个数
        # print(deepth, zoom_scaler)
        if deepth is None or zoom_scaler is None:
            return None

        num = 1 / (deepth * 10 / zoom_scaler / 734)  # 注意，734的获取方式：利用mitk或者图像软件，在刻度0的地方点击一下获得y，然后在最后一个刻度，获得坐标，然后相减即可。不用+1，因为用间隔

        return int(num + 0.5)

    def ocr_instant(self, img=None):
        # 目前只需要识别四个数值，保证实时性: skin deepth, A , B ,alpha

        # en ch

        # 实际使用的时候，需要放开以下两行
        if img is None:
            img = pyautogui.screenshot(allScreens=False, region=(0, 0, 1920, 1080))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # showimg(img)


        # time_in = time.time()

        # 2025年6月17日byYoung;需要知道是否在进行段落的测量


        # 如果在段落测量，就更新AB和Alpha的值
        self.detect_distance_in_img(img)

        results = self._ocr(img[822:944, 1304:])

        # self.results_show(img[822:944, 1304:], results)

        zoom_scaler = self.find_Zoom_Scaler_in_ocr_results(results)
        is_freeze = self.find_is_freeze_in_ocr_results(results)
        settings = self.find_other_setting_in_ocr_results(results)

        self.MEASSURE['Is_Freeze'] = is_freeze

        if zoom_scaler is not None:
            self.MEASSURE['Zoom_scaler'] = zoom_scaler
        else:
            self.MEASSURE['Zoom_scaler'] = 1.0



        # MEASSURE.update(settings)# 不能直接update，因为用户可能切换，使得不是所有的setting都能识别，如深度，但是深度已经设定好了
        for key in settings.keys():
            if settings[key] is not None:  # 设定的参数，由于有移动，没有识别到，默认为前一次的结果
                self.MEASSURE[key] = settings[key]
            else:
                pass


        points_per_mm = self.cal_points_per_mm(self.MEASSURE['深度'], self.MEASSURE['Zoom_scaler'])

        self.MEASSURE['Points_Per_MM'] = points_per_mm

        # time_in = time.time() - time_in
        # print(time_in)


    def start_ocr_server(self):
        while True:
            try:
                self.ocr_instant()
                time.sleep(self.time_skip)
            except:
                pass

    def stop_ocr_server(self):
        pass



if __name__ == '__main__':

    # for i in range(9):
    #     path = f'screensshots/{i}.bmp'
    #     img = cv2.imread(path)
    #     ocr_instant(img)

    path = f'screensshots/new1.bmp'
    img = cv2.imread(path)

    # print(img.shape)
    cv2.imshow("img1", img)
    cv2.waitKey(0)
    print(img.shape)

    time_in1 = time.time()
    ocr = OCRDetect()

    time_in2 = time.time()

    counter = 100
    while counter > 0:
        time_in3 = time.time()
        # print(time_in2 - time_in1)

        ocr.ocr_instant()

        print(time.time() - time_in3)
        print(ocr.MEASSURE)
        counter -= 1

    print(time.time() - time_in2)





