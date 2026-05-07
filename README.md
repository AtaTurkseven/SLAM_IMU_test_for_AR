# SLAM IMU Test for AR

Python utilities for collecting, preparing, and sanity-checking camera + IMU
data for visual-inertial SLAM / AR experiments.

The repository contains code only. Raw recordings, frame dumps, calibration
images, generated EuRoC datasets, and output plots are intentionally kept out of
Git because they can become very large.

## What is included

- Camera + IMU recording from a USB camera and serial IMU stream.
- Camera calibration from checkerboard images.
- Dataset trimming and timestamp repair helpers.
- Export to a EuRoC-style folder layout.
- Visual odometry smoke tests using ORB feature matching.
- Camera/IMU rotation agreement checks.

## Repository Layout

```text
.
+-- record_vio_logger.py                 # Record camera frames and IMU packets
+-- camera_calib.py                      # Robust checkerboard camera calibration
+-- export_euroc.py                      # Export repaired data to EuRoC layout
+-- vo_smoke_test.py                     # Quick ORB feature matching sanity test
+-- vo_pose_test.py                      # Monocular visual pose sanity check
+-- vio_rotation_sync_test.py            # Compare visual rotation with IMU gyro
+-- calib_check.py                       # Inspect calibration output
+-- calib_test.py                        # Calibration verification helper
+-- validate_trimmed.py                  # Validate trimmed dataset consistency
+-- take_images_calib.py                 # Capture calibration images
+-- OLD/deneme.py                        # Older experimental script
+-- test_vio_01/test_dataset.py          # Dataset inspection helper
+-- test_vio_01/trim_dataset.py          # Dataset trimming helper
+-- test_vio_01_trimmed/
    +-- repair_dataset.py
    +-- repair_using_stationary_calib.py
```

## Requirements

- Python 3.10 or newer
- OpenCV-compatible camera
- Serial IMU device that emits the packet format expected by
  `record_vio_logger.py`

Install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python numpy pandas pyserial pyyaml matplotlib
```

On Linux/macOS, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Typical Workflow

### 1. Record camera + IMU data

```bash
python record_vio_logger.py --port COM5 --camera 0 --out test_vio_01 --duration 60
```

Useful options:

- `--baud`: Serial baud rate, default `500000`
- `--width` / `--height`: Camera resolution, default `640x480`
- `--fps`: Requested camera FPS, default `30`

The recorder creates a local dataset folder like:

```text
test_vio_01/
+-- cam.csv
+-- imu.csv
+-- frames/
```

### 2. Capture calibration images

```bash
python take_images_calib.py
```

Place calibration images in a local `camera_calib_images/` folder, then run:

```bash
python camera_calib.py
```

This writes `camera_calib_robust.yaml`.

### 3. Repair or trim a dataset

```bash
python test_vio_01/trim_dataset.py
python test_vio_01_trimmed/repair_dataset.py
python validate_trimmed.py
```

### 4. Export to EuRoC-style format

```bash
python export_euroc.py
```

Expected output:

```text
euroc_test/
+-- mav0/
    +-- cam0/
    |   +-- data.csv
    |   +-- sensor.yaml
    |   +-- data/
    +-- imu0/
        +-- data.csv
        +-- sensor.yaml
```

### 5. Run sanity checks

```bash
python vo_smoke_test.py
python vo_pose_test.py
python vio_rotation_sync_test.py
```

## Data Policy

Large generated folders should stay local and should not be committed:

- `OLD/frames/`
- `camera_calib_images/`
- `euroc_test/`
- `test_vio_01/frames/`
- `test_vio_01_repaired/`
- `test_vio_01_trimmed/frames/`
- `*.csv` recordings
- generated `*.png` plots/results

If a dataset needs to be shared, use an external storage location or Git LFS
instead of normal Git history.

## Notes

- IMU scaling constants in `record_vio_logger.py` are currently set for the
  expected sensor stream. Check `ACCEL_LSB_PER_G` and `GYRO_LSB_PER_DPS` if a
  different IMU configuration is used.
- `export_euroc.py` uses approximate IMU noise parameters. Tune them before
  using the exported data with a production VIO/SLAM pipeline.
- `vio_rotation_sync_test.py` is a diagnostic tool, not a full SLAM system.
