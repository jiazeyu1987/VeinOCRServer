# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VeinOCRServer** is a Windows-only Python application that performs real-time OCR (Optical Character Recognition) on medical ultrasound screens. It extracts measurements and treatment data from ultrasound imaging systems, providing both real-time OCR capabilities and before/after image comparison for medical treatment tracking.

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
# Build Cython extensions
python setup.py build_ext --inplace
setup.bat

# Create Windows executable
pyinstaller ocrapp_pureray.spec
```

### Installation

```bash
pip install -r requirement.txt
```

### Dependency Notes

- **NumPy/Compatibility**: NumPy is pinned to 1.26.4, OpenCV to <4.12. Upgrading either may break compatibility with PaddleOCR 2.x on Windows.
- **PaddleOCR**: Keep version <3. PaddleOCR 3.x pulls in PyTorch dependencies that often fail on Windows due to DLL conflicts.
- **GPU vs CPU**: The `main.py` entry point forces CPU mode by overriding `settings["GPU"] = False`. To use GPU, modify the override or run `server.py` directly.

## Architecture

### Core Components

**ImageProcessServer** (`server.py`)
- TCP server on port 30415 (configurable)
- Manages OCR detection and image comparison threads
- Provides password-protected API (password: "31415")
- Request types: ONLINE (real-time OCR), OFFLINE (batch image comparison), OPENOCR/CLOSEOCR (continuous OCR)

**OCRDetect** (`ocr_detect.py`)
- PaddleOCR wrapper for medical text recognition
- Extracts ultrasound measurements: SkinDepth, A, B, Alpha, Depth, IsFreeze, Points_Per_MM
- Supports CPU and GPU processing
- Model weights stored in `./whl/` directory

**ComparePoints** (`treat_compare_img.py`)
- Treatment before/after image comparison
- SQLite database integration for storing results
- Thread-safe image processing

### Request/Response Protocol

The server uses a simple TCP protocol with JSON payloads:

- **ONLINE**: Returns real-time OCR results as JSON
  ```json
  {"SkinDepth": 5.2, "A": 4.3, "B": 3.1, "Alpha": 0, "Depth": 45, "IsFreeze": false, "Points_Per_MM": 10}
  ```

- **OFFLINE**: Batch image comparison with point tracking
  - Parameters: `{"point_id": 123, "time_out": 100, "is_save": true}`
  - Start: send unique point_id
  - Stop: send same point_id again
  - Response: `None` (by design)

- **OPENOCR/CLOSEOCR**: Control continuous OCR recognition

### Client Applications

- `test_ocr_client.py`: Full-featured Tkinter GUI client
  - Watch mode for continuous ONLINE requests
  - Configurable request intervals
  - Integration with image comparison
- `client.py`: Basic CLI client for testing
- `treat_compare_img.py --gui`: Standalone control panel

### Configuration

The `settings` file (JSON) controls:
```json
{
  "GPU": true,           // Use GPU if available (overridden by main.py)
  "width_x": 2,          // Grid width for image processing
  "height_y": 4,         // Grid height for image processing
  "binary_threshold": 10,// Image difference threshold
  "drawcontour": true,   // Draw contours on processed images
  "if_align": false      // Enable image alignment
}
```

## Environment Setup Notes

### Windows-Specific Configuration

The application requires these environment variables (set in `main.py` and `server.py`):
- `KMP_DUPLICATE_LIB_OK=TRUE`: Workaround for OpenMP runtime conflicts between MKL, PaddlePaddle, and OpenCV
- `OMP_NUM_THREADS=1`: Limits OpenMP threads to avoid conflicts
- `CUDA_VISIBLE_DEVICES=""` and `FLAGS_use_cuda="0"`: Force CPU mode when set

### Logging

Logs are written to `ocrlog/ocrapp_YYYY-MM-DD.log` with daily rotation. All components use the same logging framework with INFO level by default.

### Data Storage

- Screenshots: `screenshots/` directory
- Comparison images: `D:/software_data/imgs/`
- Database: SQLite (path configurable in code)

## Medical Context

This application is specifically designed for medical ultrasound systems. The OCR targets:
- Measurements (depths A/B, angles, skin distance)
- Treatment status (freeze state)
- Scale information (points per millimeter)
- System settings (gain, frequency, image enhancement)

The terminology and UI layout expectations are specific to medical ultrasound devices.
