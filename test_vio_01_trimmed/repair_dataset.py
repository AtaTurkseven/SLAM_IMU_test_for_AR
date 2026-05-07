import pandas as pd
import numpy as np
from pathlib import Path
import shutil

DATASET = Path("test_vio_01_trimmed")
OUT = Path("test_vio_01_repaired")

G = 9.80665
GYRO_LSB_PER_DPS = 131.0

imu = pd.read_csv(DATASET / "imu.csv")
cam = pd.read_csv(DATASET / "cam.csv")

OUT.mkdir(exist_ok=True)

# Use PC host time only to select first/last still parts.
t_host = imu["host_time_ns"].to_numpy() * 1e-9
t_host = t_host - t_host[0]
duration = t_host[-1]

# Assumes your recording had still sections at beginning and end.
# Use first 8s and last 8s as stationary calibration windows.
still_mask = (t_host < 8.0) | (t_host > duration - 8.0)

raw_ax = imu["raw_ax"].to_numpy()
raw_ay = imu["raw_ay"].to_numpy()
raw_az = imu["raw_az"].to_numpy()

raw_gx = imu["raw_gx"].to_numpy()
raw_gy = imu["raw_gy"].to_numpy()
raw_gz = imu["raw_gz"].to_numpy()

raw_acc_norm = np.sqrt(raw_ax**2 + raw_ay**2 + raw_az**2)

ACCEL_LSB_PER_G = np.median(raw_acc_norm[still_mask])

gyro_bias_x = np.mean(raw_gx[still_mask])
gyro_bias_y = np.mean(raw_gy[still_mask])
gyro_bias_z = np.mean(raw_gz[still_mask])

print("Estimated ACCEL_LSB_PER_G:", ACCEL_LSB_PER_G)
print("Gyro raw bias:")
print("gx:", gyro_bias_x)
print("gy:", gyro_bias_y)
print("gz:", gyro_bias_z)

# Recompute acceleration
imu["ax_mps2"] = raw_ax / ACCEL_LSB_PER_G * G
imu["ay_mps2"] = raw_ay / ACCEL_LSB_PER_G * G
imu["az_mps2"] = raw_az / ACCEL_LSB_PER_G * G

# Recompute gyro with bias subtraction
imu["gx_rads"] = (raw_gx - gyro_bias_x) / GYRO_LSB_PER_DPS * np.pi / 180.0
imu["gy_rads"] = (raw_gy - gyro_bias_y) / GYRO_LSB_PER_DPS * np.pi / 180.0
imu["gz_rads"] = (raw_gz - gyro_bias_z) / GYRO_LSB_PER_DPS * np.pi / 180.0

# Build a synced timestamp for IMU samples.
# Fit Arduino imu_time_us -> PC host_time_ns.
imu_time_us = imu["imu_time_us"].to_numpy().astype(np.float64)
host_time_ns = imu["host_time_ns"].to_numpy().astype(np.float64)

a, b = np.polyfit(imu_time_us, host_time_ns, 1)
imu["timestamp_ns"] = np.round(a * imu_time_us + b).astype(np.int64)

# Camera timestamp already uses host_time_ns.
cam["timestamp_ns"] = cam["host_time_ns"].astype(np.int64)

imu.to_csv(OUT / "imu.csv", index=False)
cam.to_csv(OUT / "cam.csv", index=False)

# Do not duplicate all images unless needed.
# Just point to the original frames folder for now.
print("Saved repaired imu/cam CSV to:", OUT)

# Quick validation
acc_norm = np.sqrt(
    imu["ax_mps2"]**2 +
    imu["ay_mps2"]**2 +
    imu["az_mps2"]**2
)

gyro_norm = np.sqrt(
    imu["gx_rads"]**2 +
    imu["gy_rads"]**2 +
    imu["gz_rads"]**2
)

still_acc = acc_norm[still_mask]
still_gyro = gyro_norm[still_mask]

print()
print("Still accel norm mean:", still_acc.mean())
print("Still accel norm std:", still_acc.std())
print("Still gyro norm mean:", still_gyro.mean())
print("Still gyro norm std:", still_gyro.std())

t_imu = imu["timestamp_ns"].to_numpy() * 1e-9
t_imu -= t_imu[0]
dt_imu = np.diff(t_imu)

print()
print("IMU mapped median rate:", 1.0 / np.median(dt_imu))
print("IMU mapped mean rate:", len(imu) / t_imu[-1])
print("IMU mapped dt max:", dt_imu.max())