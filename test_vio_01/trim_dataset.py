import pandas as pd
from pathlib import Path
import shutil

DATASET = Path("test_vio_01")
OUT = Path("test_vio_01_trimmed")

imu = pd.read_csv(DATASET / "imu.csv")
cam = pd.read_csv(DATASET / "cam.csv")

cam_start = cam["host_time_ns"].iloc[0]
cam_end = cam["host_time_ns"].iloc[-1]

imu_trim = imu[
    (imu["host_time_ns"] >= cam_start) &
    (imu["host_time_ns"] <= cam_end)
].copy()

OUT.mkdir(exist_ok=True)
(OUT / "frames").mkdir(exist_ok=True)

imu_trim.to_csv(OUT / "imu.csv", index=False)
cam.to_csv(OUT / "cam.csv", index=False)

for filename in cam["filename"]:
    shutil.copy(DATASET / "frames" / filename, OUT / "frames" / filename)

print("Original IMU rows:", len(imu))
print("Trimmed IMU rows:", len(imu_trim))
print("Camera frames:", len(cam))

t_imu = imu_trim["host_time_ns"].to_numpy() * 1e-9
t_imu -= t_imu[0]

t_cam = cam["host_time_ns"].to_numpy() * 1e-9
t_cam -= t_cam[0]

print("Trimmed IMU duration:", t_imu[-1])
print("Camera duration:", t_cam[-1])
print("Saved:", OUT)