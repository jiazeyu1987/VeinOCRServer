# encoding=utf-8
"""
Minimal peak detection logic extracted from:
`D:\\ProjectPackage\\SimpleFEM\\fem_refactor\\external\\peak_detection.py`

Purpose in this repo:
- Provide `detect_peaks()` for VeinOCRServer without depending on the SimpleFEM
  source tree or absolute imports.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# Optional external "improved" implementation (kept for compatibility with the
# upstream API; disabled by default).
try:
    from improved_peak_detection import detect_peaks_improved  # type: ignore

    IMPROVED_DETECTION_AVAILABLE = True
except Exception:
    IMPROVED_DETECTION_AVAILABLE = False

USE_IMPROVED_DETECTION: bool = False


def calculate_frame_difference(
    curve: List[float],
    peak_start: int,
    peak_end: int,
    avgFrames: int = 5,
) -> float:
    """
    frameDifference = average(curve[peak_end+1 : peak_end+1+avgFrames])
                      - average(curve[peak_start-avgFrames : peak_start])
    """
    n = len(curve)
    if n == 0:
        return 0.0

    frame_count = int(avgFrames)
    if frame_count <= 0:
        frame_count = 5

    before_start = max(0, peak_start - frame_count)
    before_end = max(0, peak_start - 1)
    if before_start <= before_end:
        before_vals = curve[before_start : before_end + 1]
        before_avg = sum(before_vals) / len(before_vals)
    else:
        before_avg = curve[peak_start] if 0 <= peak_start < n else 0.0

    after_start = min(n - 1, peak_end + 1)
    after_end = min(n - 1, peak_end + frame_count)
    if after_start <= after_end:
        after_vals = curve[after_start : after_end + 1]
        after_avg = sum(after_vals) / len(after_vals)
    else:
        after_avg = curve[peak_end] if 0 <= peak_end < n else 0.0

    return float(after_avg - before_avg)


def classify_peak_color(frameDifference: float, differenceThreshold: float = 0.5) -> str:
    return "green" if frameDifference >= differenceThreshold else "red"


def detect_white_peaks_by_threshold_improved(
    curve: List[float],
    *,
    threshold: float,
    marginFrames: int = 5,
    silenceFrames: int = 0,
    differenceThreshold: float = 0.5,
    avgFrames: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Detect contiguous segments where curve[i] >= threshold, apply:
    - marginFrames: minimal spacing between peaks (keep higher max if overlap)
    - silenceFrames: require X frames below threshold around each peak
    Then compute frameDifference for color classification.
    """
    n = len(curve)
    if n == 0:
        return []

    raw: List[Tuple[int, int]] = []
    in_peak = False
    start = 0
    for i, v in enumerate(curve):
        if v >= threshold:
            if not in_peak:
                start = i
                in_peak = True
        else:
            if in_peak:
                raw.append((start, i - 1))
                in_peak = False
    if in_peak:
        raw.append((start, n - 1))

    if not raw:
        return []

    if marginFrames > 0 and len(raw) > 1:
        filtered: List[Tuple[int, int]] = [raw[0]]
        for s, e in raw[1:]:
            last_s, last_e = filtered[-1]
            spacing = s - last_e
            if spacing >= marginFrames:
                filtered.append((s, e))
            else:
                last_max = max(curve[last_s : last_e + 1])
                cur_max = max(curve[s : e + 1])
                if cur_max > last_max:
                    filtered[-1] = (s, e)
        raw = filtered

    if silenceFrames > 0 and raw:
        silenced: List[Tuple[int, int]] = []
        for s, e in raw:
            if s - silenceFrames < 0 or e + silenceFrames >= n:
                continue
            pre_ok = all(curve[i] < threshold for i in range(s - silenceFrames, s))
            post_ok = all(curve[i] < threshold for i in range(e + 1, e + 1 + silenceFrames))
            if pre_ok and post_ok:
                silenced.append((s, e))
        raw = silenced

    if not raw:
        return []

    result: List[Tuple[int, int, float]] = []
    for s, e in raw:
        frame_diff = calculate_frame_difference(curve, s, e, avgFrames=avgFrames)
        if abs(frame_diff) > 15.0:
            continue
        result.append((s, e, frame_diff))
    return result


def detect_peaks(
    curve: List[float],
    *,
    threshold: float = 105.0,
    marginFrames: int = 5,
    differenceThreshold: float = 0.5,
    silenceFrames: int = 0,
    avgFrames: int = 5,
    use_improved: bool = False,
    **config_params,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Returns:
      (green_peaks, red_peaks) where each is a list of (start, end) indices.
    """
    if not curve:
        return [], []

    use_improved_algo = use_improved or USE_IMPROVED_DETECTION
    if use_improved_algo and IMPROVED_DETECTION_AVAILABLE:
        if "silenceFrames" not in config_params:
            config_params["silenceFrames"] = silenceFrames
        if "avgFrames" not in config_params:
            config_params["avgFrames"] = avgFrames
        return detect_peaks_improved(
            curve,
            threshold,
            marginFrames,
            differenceThreshold,
            **config_params,
        )

    peaks_with_diff = detect_white_peaks_by_threshold_improved(
        curve,
        threshold=threshold,
        marginFrames=marginFrames,
        silenceFrames=silenceFrames,
        differenceThreshold=differenceThreshold,
        avgFrames=avgFrames,
    )

    green_peaks: List[Tuple[int, int]] = []
    red_peaks: List[Tuple[int, int]] = []
    for start, end, frame_diff in peaks_with_diff:
        color = classify_peak_color(frame_diff, differenceThreshold)
        if color == "green":
            green_peaks.append((start, end))
        else:
            red_peaks.append((start, end))
    return green_peaks, red_peaks

