import pandas as pd
from pathlib import Path
import shutil
import yaml

BASE = Path(__file__).resolve().parent

REPAIRED_DATASET = BASE / "test_vio_01_repaired"
FRAMES_SOURCE = BASE / "test_vio_01_trimmed" / "frames"
CALIB_FILE = BASE / "test_vio_01_repaired" / "camera_calib_robust.yaml"

OUT = BASE / "euroc_test"
CAM_OUT = OUT / "mav0" / "cam0"
IMU_OUT = OUT / "mav0" / "imu0"

(CAM_OUT / "data").mkdir(parents=True, exist_ok=True)
IMU_OUT.mkdir(parents=True, exist_ok=True)

imu = pd.read_csv(REPAIRED_DATASET / "imu.csv")
cam = pd.read_csv(REPAIRED_DATASET / "cam.csv")

with open(CALIB_FILE, "r") as f:
    calib = yaml.safe_load(f)

# -------------------------
# Camera data.csv
# -------------------------
cam_csv_path = CAM_OUT / "data.csv"

with open(cam_csv_path, "w") as f:
    f.write("#timestamp [ns],filename\n")

    for _, row in cam.iterrows():
        ts = int(row["timestamp_ns"])
        old_name = row["filename"]
        new_name = f"{ts}.png"

        src = FRAMES_SOURCE / old_name
        dst = CAM_OUT / "data" / new_name

        if src.exists():
            shutil.copy(src, dst)
            f.write(f"{ts},{new_name}\n")
        else:
            print("Missing frame:", src)

# -------------------------
# IMU data.csv
# -------------------------
imu_csv_path = IMU_OUT / "data.csv"

with open(imu_csv_path, "w") as f:
    f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")

    for _, row in imu.iterrows():
        ts = int(row["timestamp_ns"])

        gx = row["gx_rads"]
        gy = row["gy_rads"]
        gz = row["gz_rads"]

        ax = row["ax_mps2"]
        ay = row["ay_mps2"]
        az = row["az_mps2"]

        f.write(f"{ts},{gx},{gy},{gz},{ax},{ay},{az}\n")

# -------------------------
# Camera sensor.yaml
# -------------------------
fx = calib["camera_matrix"]["fx"]
fy = calib["camera_matrix"]["fy"]
cx = calib["camera_matrix"]["cx"]
cy = calib["camera_matrix"]["cy"]

d = calib["distortion_coefficients"]

cam_sensor = {
    "sensor_type": "camera",
    "comment": "monocular camera calibration",
    "T_BS": {
        "rows": 4,
        "cols": 4,
        "data": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    },
    "rate_hz": 30,
    "resolution": [
        calib["image_width"],
        calib["image_height"],
    ],
    "camera_model": "pinhole",
    "intrinsics": [fx, fy, cx, cy],
    "distortion_model": "radial-tangential",
    "distortion_coefficients": [
        d["k1"],
        d["k2"],
        d["p1"],
        d["p2"],
    ],
}

with open(CAM_OUT / "sensor.yaml", "w") as f:
    yaml.dump(cam_sensor, f, sort_keys=False)

# -------------------------
# IMU sensor.yaml
# -------------------------
imu_sensor = {
    "sensor_type": "imu",
    "comment": "MPU6050 approximate noise values; tune later",
    "T_BS": {
        "rows": 4,
        "cols": 4,
        "data": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    },
    "rate_hz": 100,
    "gyroscope_noise_density": 0.01,
    "gyroscope_random_walk": 0.0002,
    "accelerometer_noise_density": 0.1,
    "accelerometer_random_walk": 0.002,
}

with open(IMU_OUT / "sensor.yaml", "w") as f:
    yaml.dump(imu_sensor, f, sort_keys=False)

print("Exported EuRoC-style dataset to:", OUT)
print("Camera frames:", len(cam))
print("IMU samples:", len(imu))