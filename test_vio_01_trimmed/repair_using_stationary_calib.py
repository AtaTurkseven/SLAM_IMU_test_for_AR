import pandas as pd
import numpy as np
from pathlib import Path

# Change these if needed
STATIONARY_IMU = Path("imu_only.csv")
DATASET = Path(".")
OUT = Path("../test_vio_01_repaired")

G = 9.80665
GYRO_LSB_PER_DPS = 131.0

# -------------------------
# 1. Load stationary IMU calibration file
# -------------------------
calib = pd.read_csv(STATIONARY_IMU)

cax = calib["raw_ax"].to_numpy()
cay = calib["raw_ay"].to_numpy()
caz = calib["raw_az"].to_numpy()

cgx = calib["raw_gx"].to_numpy()
cgy = calib["raw_gy"].to_numpy()
cgz = calib["raw_gz"].to_numpy()

raw_acc_norm = np.sqrt(cax**2 + cay**2 + caz**2)

# Remove obvious spike samples before estimating calibration
median_norm = np.median(raw_acc_norm)
good = np.abs(raw_acc_norm - median_norm) < 1000

ACCEL_LSB_PER_G = np.median(raw_acc_norm[good])

gyro_bias_x = np.mean(cgx[good])
gyro_bias_y = np.mean(cgy[good])
gyro_bias_z = np.mean(cgz[good])

print("Calibration from stationary file")
print("ACCEL_LSB_PER_G:", ACCEL_LSB_PER_G)
print("Gyro raw bias:")
print("gx:", gyro_bias_x)
print("gy:", gyro_bias_y)
print("gz:", gyro_bias_z)

# Validate stationary calibration
calib_ax = cax / ACCEL_LSB_PER_G * G
calib_ay = cay / ACCEL_LSB_PER_G * G
calib_az = caz / ACCEL_LSB_PER_G * G

calib_gx = (cgx - gyro_bias_x) / GYRO_LSB_PER_DPS * np.pi / 180.0
calib_gy = (cgy - gyro_bias_y) / GYRO_LSB_PER_DPS * np.pi / 180.0
calib_gz = (cgz - gyro_bias_z) / GYRO_LSB_PER_DPS * np.pi / 180.0

calib_acc_norm = np.sqrt(calib_ax**2 + calib_ay**2 + calib_az**2)
calib_gyro_norm = np.sqrt(calib_gx**2 + calib_gy**2 + calib_gz**2)

print()
print("Stationary validation")
print("accel norm mean:", calib_acc_norm[good].mean())
print("accel norm std:", calib_acc_norm[good].std())
print("gyro norm mean:", calib_gyro_norm[good].mean())
print("gyro norm std:", calib_gyro_norm[good].std())

# -------------------------
# 2. Repair moving camera+IMU dataset
# -------------------------
imu = pd.read_csv(DATASET / "imu.csv")
cam = pd.read_csv(DATASET / "cam.csv")

raw_ax = imu["raw_ax"].to_numpy()
raw_ay = imu["raw_ay"].to_numpy()
raw_az = imu["raw_az"].to_numpy()

raw_gx = imu["raw_gx"].to_numpy()
raw_gy = imu["raw_gy"].to_numpy()
raw_gz = imu["raw_gz"].to_numpy()

imu["ax_mps2"] = raw_ax / ACCEL_LSB_PER_G * G
imu["ay_mps2"] = raw_ay / ACCEL_LSB_PER_G * G
imu["az_mps2"] = raw_az / ACCEL_LSB_PER_G * G

imu["gx_rads"] = (raw_gx - gyro_bias_x) / GYRO_LSB_PER_DPS * np.pi / 180.0
imu["gy_rads"] = (raw_gy - gyro_bias_y) / GYRO_LSB_PER_DPS * np.pi / 180.0
imu["gz_rads"] = (raw_gz - gyro_bias_z) / GYRO_LSB_PER_DPS * np.pi / 180.0

# Fit Arduino IMU time to PC host time for synced timestamps
imu_time_us = imu["imu_time_us"].to_numpy().astype(np.float64)
host_time_ns = imu["host_time_ns"].to_numpy().astype(np.float64)

a, b = np.polyfit(imu_time_us, host_time_ns, 1)

imu["timestamp_ns"] = np.round(a * imu_time_us + b).astype(np.int64)
cam["timestamp_ns"] = cam["host_time_ns"].astype(np.int64)

OUT.mkdir(exist_ok=True)

imu.to_csv(OUT / "imu.csv", index=False)
cam.to_csv(OUT / "cam.csv", index=False)

print()
print("Saved repaired dataset to:", OUT)

# -------------------------
# 3. Validate repaired moving dataset timing
# -------------------------
t_imu = imu["timestamp_ns"].to_numpy() * 1e-9
t_imu -= t_imu[0]
dt_imu = np.diff(t_imu)

t_cam = cam["timestamp_ns"].to_numpy() * 1e-9
t_cam -= t_cam[0]
dt_cam = np.diff(t_cam)

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

print()
print("Moving dataset validation")
print("IMU rows:", len(imu))
print("Camera frames:", len(cam))
print("IMU duration:", t_imu[-1])
print("IMU median rate:", 1.0 / np.median(dt_imu))
print("IMU mean rate:", len(imu) / t_imu[-1])
print("IMU dt max:", dt_imu.max())
print("Camera mean FPS:", len(cam) / t_cam[-1])
print("Camera dt max:", dt_cam.max())

print()
print("Full motion accel norm mean:", acc_norm.mean())
print("Full motion accel norm std:", acc_norm.std())
print("Full motion gyro norm mean:", gyro_norm.mean())
print("Full motion gyro norm std:", gyro_norm.std())