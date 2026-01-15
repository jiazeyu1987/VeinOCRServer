# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VeinOCRServer** is a Windows-only Python application that performs real-time OCR (Optical Character Recognition) on medical ultrasound screens. It extracts measurements and treatment data from ultrasound imaging systems, providing both real-time OCR capabilities and before/after image comparison for medical treatment tracking.

## Important Windows Constraints

This application **only runs on Windows** due to:
- PyAutoGUI screen capture (`pyautogui.screenshot()`)
- Windows-specific DLL dependencies (PaddlePaddle, OpenCV)
- Hardcoded screen region coordinates (1920x1080 ultrasound display area)
- SQLite database paths using Windows-style paths (`D:/software_data/`)

## Common Commands

### Running the Application

```bash
# Start the main OCR server (always runs in CPU mode via main.py override)
python main.py

# Start the full-featured GUI client
python test_ocr_client.py

# Start the CLI client
python test_ocr_client.py --cli

# Start standalone GUI control panel for OCR + image comparison
python treat_compare_img.py --gui
```

### Testing

```bash
# Test PaddleOCR with GPU
python test_paddleocr_gpu.py -gpu

# Test PaddleOCR with CPU
python test_paddleocr_gpu.py -cpu
```

### Building

```bash
# Build Cython extensions (optional - for performance optimization)
python setup.py build_ext --inplace
# Or use the batch wrapper
setup.bat

# Create Windows executable using PyInstaller
pyinstaller ocrapp_pureray.spec
# Output: dist/ocrapp_pureray.exe (single-file executable)
# Note: The spec file bundles all dependencies including PaddleOCR models
```

### Installation

```bash
pip install -r requirement.txt
```

### Dependency Notes

- **NumPy/Compatibility**: NumPy is pinned to 1.26.4, OpenCV to <4.12. Upgrading either may break compatibility with PaddleOCR 2.x on Windows due to ABI mismatches.
- **PaddleOCR**: Keep version <3. PaddleOCR 3.x pulls in PaddleX/ModelScope/PyTorch dependencies that often fail on Windows due to DLL conflicts.
- **GPU vs CPU**: The `main.py` entry point forces CPU mode by monkeypatching `server.ImageProcessServer.load_setting`. To use GPU:
  - Run `server.py` directly instead of `main.py`, or
  - Remove the override in `main.py:36-43`
  - Replace `paddlepaddle` with appropriate `paddlepaddle-gpu` wheel for your CUDA version in requirement.txt
- **SciPy/scikit-image**: Pinned to versions compatible with NumPy 1.x (scipy<1.12, scikit-image<0.23)
- **Shapely**: Pinned to <2 to avoid NumPy 2.x ABI issues
- **Python Version**: Python 3.10 or 3.11 recommended

## Architecture

### Core Components

**ImageProcessServer** (`server.py`)
- TCP server on port 30415 (configurable)
- Manages OCR detection thread and image comparison threads
- Provides password-protected API (password: "31415")
- Request types: ONLINE (real-time OCR), OFFLINE (batch image comparison), CLOSE (legacy)
- Uses `threading.Thread` for both OCR and comparison tasks
- IMPORTANT: OFFLINE responses are intentionally set to `None` (server.py:248) to avoid packet fragmentation when rapid ONLINE/OFFLINE requests are sent

**OCRDetect** (`ocr_detect.py`)
- PaddleOCR wrapper for medical text recognition
- Runs continuous OCR loop in daemon thread, updating `self.MEASSURE` dict
- Extracts ultrasound measurements: SkinDepth, A, B, Alpha, Depth, IsFreeze, Points_Per_MM
- Supports CPU and GPU processing (GPU requires CUDA-capable NVIDIA GPU)
- Model weights stored in `./whl/det/`, `./whl/rec/`, `./whl/cls/` directories (can be customized via settings)
- Uses hardcoded screen coordinates for ultrasound display region detection
- Can be compiled to Cython extension via setup.py for performance

**ComparePoints** (`treat_compare_img.py`)
- Treatment before/after image comparison using grayscale detection
- SQLite database integration for storing results (writes to both `ccwssm` and `zccwssm` databases)
- Thread-safe image processing using `threading.Event` for stop signaling
- Detects image changes based on grayscale value in ultrasound display region
- Stores images to `D:/software_data/imgs/` (hardcoded path)
- Can be compiled to Cython extension via setup.py for performance
- Has both direct test mode and standalone GUI mode (`--gui` flag)

### Request/Response Protocol

The server uses a simple TCP protocol with JSON payloads:

**Request Format**: `TYPE;PASSWORD;ARGUMENT`
- Example: `ONLINE;31415;` (no argument needed for ONLINE)
- Example: `OFFLINE;31415;{"point_id":123,"time_out":100,"is_save":true}`

**Request Types**:
- **ONLINE**: Returns real-time OCR results as JSON
  ```json
  {"SkinDepth": 5.2, "A": 4.3, "B": 3.1, "Alpha": 0, "Depth": 45, "IsFreeze": false, "Points_Per_MM": 10}
  ```

- **OFFLINE**: Batch image comparison with point tracking
  - Parameters: `{"point_id": 123, "time_out": 100, "is_save": true}`
  - Start: send unique point_id (spawns comparison thread)
  - Stop: send same point_id again (sets stop event, joins thread)
  - Response: `None` (by design - see server.py:248)

- **CLOSE**: Legacy command (no-op)

### Client Applications

- `test_ocr_client.py`: Full-featured Tkinter GUI client
  - Watch mode for continuous ONLINE requests
  - Configurable request intervals
  - Integration with image comparison (combined OCR+Compare buttons)
  - CLI mode available via `--cli` flag
  - See TEST_CLIENT_GUIDE.md for detailed usage

- `client.py`: Basic CLI client for testing

- `treat_compare_img.py --gui`: Standalone GUI control panel for OCR + image comparison
  - Independent TCP client to ImageProcessServer
  - Provides start/stop controls for both OCR and comparison tasks
  - See GUI_CONTROL_README.md (Chinese) for detailed usage

- `treat_compare_img.py` (without `--gui`): Direct test mode for ComparePoints class

### Configuration

The `settings` file (JSON) in the project root controls:
```json
{
  "GPU": true,           // Use GPU if available (overridden by main.py)
  "width_x": 2,          // Grid width for image comparison processing
  "height_y": 4,         // Grid height for image comparison processing
  "binary_threshold": 10,// Image difference threshold for comparison
  "drawcontour": true,   // Draw contours on processed comparison images
  "if_align": false,     // Enable image alignment in comparison
  "det": "./whl/det/ch/ch_PP-OCRv4_det_infer",  // Custom detection model path
  "rec": "./whl/rec/ch/ch_PP-OCRv4_rec_infer",  // Custom recognition model path
  "cls": "./whl/cls/ch_ppocr_mobile_v2.0_cls_infer",  // Custom classification model path
  "time_skip": 0,        // Sleep interval between OCR frames (seconds)
  "log": false           // Show PaddleOCR logs
}
```
If `settings` file doesn't exist, defaults are used.

## Environment Setup Notes

### Windows-Specific Configuration

The application requires these environment variables (set at top of `main.py` and `server.py`):
- `KMP_DUPLICATE_LIB_OK=TRUE`: Workaround for OpenMP runtime conflicts between MKL, PaddlePaddle, and OpenCV
- `OMP_NUM_THREADS=1`: Limits OpenMP threads to avoid conflicts
- `CUDA_VISIBLE_DEVICES=""` and `FLAGS_use_cuda="0"`: Force CPU mode when set (only in main.py)

**Critical**: These must be set **before** importing numpy/paddle/opencv.

### Logging

Logs are written to `ocrlog/ocrapp_YYYY-MM-DD.log` with daily rotation. All components use the same logging framework with INFO level by default.

### Data Storage

- Screenshots: `screenshots/` directory (debug/testing)
- Comparison images: `D:/software_data/imgs/` (hardcoded in treat_compare_img.py)
- Database: SQLite at `D:/software_data/ccwssm` and `D:/software_data/zccwssm` (hardcoded)
- Logs: `ocrlog/ocrapp_YYYY-MM-DD.log`
- OCR Models: `./whl/` directory (detection, recognition, classification models)
  - Must be preserved when building executable (included in ocrapp_pureray.spec)
  - Can be customized via `settings["det"]`, `settings["rec"]`, `settings["cls"]`

**Note**: Many paths are hardcoded with forward slashes (Windows-compatible).

## Medical Context

This application is specifically designed for medical ultrasound systems (Chison/Sidoc brand devices based on terminology). The OCR targets:
- Measurements (depths A/B, Alpha angle, skin distance)
- Treatment status (IsFreeze - indicates freeze/harmonic imaging mode)
- Scale information (Points_Per_MM for pixel-to-mm conversion)
- System settings (gain, depth, frequency, image enhancement, zoom scaler)

**Screen Coordinate Assumptions**:
- Primary display: 1920x1080
- Ultrasound region: right side of screen (column 1269-1920)
- OCR crop regions hardcoded in `ocr_detect.py`:
  - A/B measurements: rows 152-217, cols 1555+
  - Settings panel: rows 822-944, cols 1304+

The terminology and UI layout expectations are specific to these medical ultrasound devices.

## Threading Model

The application uses multiple threads:

1. **Main Thread**: TCP server accept loop (`ImageProcessServer.start_server`)
2. **Client Handler Thread**: Per-client connection handling (`handle_client`)
3. **OCR Thread**: Continuous OCR detection loop (`OCRDetect.start_ocr_server`)
   - Daemon thread, runs continuously after server starts
   - Updates `OCRDetect.MEASSURE` dict in-place
   - Sleeps for `time_skip` seconds between iterations
4. **Comparison Thread**: Image comparison monitoring (`ComparePoints.detect`)
   - Spawned per OFFLINE request with unique point_id
   - Monitors grayscale changes in ultrasound region
   - Stores before/after images when change detected
   - Stopped by sending same point_id again

All threads use `threading.Event` for graceful shutdown signaling.

## Entry Point Differences

**main.py** (Production entry):
- Forces CPU mode via monkeypatch
- Sets CUDA environment variables to disable GPU
- Preferred for production deployments

**server.py** (Direct entry):
- Respects `settings["GPU"]` configuration
- Allows GPU usage if configured
- Better for development/testing with GPU

## Additional Files

- `image_difference.py`: Image comparison algorithms (contour detection, alignment)
- `compareImages/`: Alternative comparison implementations (optical flow)
- `UDP.py`: UDP communication utilities (unused in current codebase)
- `experiment_*.py`: Experimental/testing scripts
- `batchForGrabUltrasoundImage.py`: Batch screenshot tool
- `setup.py`: Cython build configuration (optional, for performance optimization)
  - Compiles: treat_compare_img.py, server.py, ocr_detect.py, pynvml.py, image_difference.py
  - Not required for development or production use
  - Use `python setup.py build_ext --inplace` to compile
- `ocrapp_pureray.spec`: PyInstaller specification for creating standalone Windows executable
  - Bundles all dependencies including PaddleOCR models from `./whl/`
  - Creates single-file executable at `dist/ocrapp_pureray.exe`