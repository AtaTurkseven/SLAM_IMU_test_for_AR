import pandas as pd
import numpy as np

DATASET = "test_vio_01_trimmed"

imu = pd.read_csv(f"{DATASET}/imu.csv")
cam = pd.read_csv(f"{DATASET}/cam.csv")

t_imu = imu["host_time_ns"].to_numpy() * 1e-9
t_imu -= t_imu[0]
dt_imu = np.diff(t_imu)

t_cam = cam["host_time_ns"].to_numpy() * 1e-9
t_cam -= t_cam[0]
dt_cam = np.diff(t_cam)

print("IMU rows:", len(imu))
print("Camera frames:", len(cam))

print()
print("IMU duration:", t_imu[-1])
print("IMU median rate:", 1.0 / np.median(dt_imu))
print("IMU mean rate:", len(imu) / t_imu[-1])
print("IMU dt max:", dt_imu.max())
print("IMU gaps > 50ms:", np.sum(dt_imu > 0.05))

print()
print("Camera duration:", t_cam[-1])
print("Camera median FPS:", 1.0 / np.median(dt_cam))
print("Camera mean FPS:", len(cam) / t_cam[-1])
print("Camera dt max:", dt_cam.max())

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
print("Accel norm mean:", acc_norm.mean())
print("Accel norm std:", acc_norm.std())
print("Gyro norm mean:", gyro_norm.mean())
print("Gyro norm std:", gyro_norm.std())