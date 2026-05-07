import pandas as pd
import numpy as np

imu = pd.read_csv("test_vio_01/imu.csv")
cam = pd.read_csv("test_vio_01/cam.csv")

t_imu = imu["imu_time_us"].to_numpy() * 1e-6
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

packet_id = imu["packet_id"].to_numpy()
missing = np.diff(packet_id) - 1

print()
print("Missing IMU packets:", missing[missing > 0].sum())
print("Packet jumps:", np.sum(missing != 0))