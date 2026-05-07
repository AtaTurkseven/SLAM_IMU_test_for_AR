import cv2
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent

CAM_CSV = BASE / "euroc_test" / "mav0" / "cam0" / "data.csv"
IMG_DIR = BASE / "euroc_test" / "mav0" / "cam0" / "data"
IMU_CSV = BASE / "euroc_test" / "mav0" / "imu0" / "data.csv"

CALIB_FILE = BASE / "test_vio_01_repaired" / "camera_calib_robust.yaml"

with open(CALIB_FILE, "r") as f:
    calib = yaml.safe_load(f)

fx = calib["camera_matrix"]["fx"]
fy = calib["camera_matrix"]["fy"]
cx = calib["camera_matrix"]["cx"]
cy = calib["camera_matrix"]["cy"]

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

d = calib["distortion_coefficients"]
dist = np.array([
    d["k1"],
    d["k2"],
    d["p1"],
    d["p2"],
    d["k3"]
], dtype=np.float64)

cam = pd.read_csv(
    CAM_CSV,
    comment="#",
    header=None,
    names=["timestamp_ns", "filename"]
)

imu = pd.read_csv(
    IMU_CSV,
    comment="#",
    header=None,
    names=[
        "timestamp_ns",
        "gx", "gy", "gz",
        "ax", "ay", "az"
    ]
)

orb = cv2.ORB_create(
    nfeatures=2500,
    fastThreshold=12
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

visual_angles = []
imu_angles = []
times = []

prev_gray = None
prev_kp = None
prev_des = None
prev_ts = None

for i, row in cam.iterrows():
    ts = int(row["timestamp_ns"])
    img_path = IMG_DIR / row["filename"]

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.undistort(gray_raw, K, dist)

    kp, des = orb.detectAndCompute(gray, None)

    if prev_gray is not None and prev_des is not None and des is not None:
        matches = matcher.knnMatch(prev_des, des, k=2)

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) >= 80:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp[m.trainIdx].pt for m in good])

            E, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if E is not None:
                _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

                # Rotation angle from visual pose
                trace = np.trace(R)
                angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
                visual_angle = float(angle)

                # IMU gyro integration between camera frames
                mask_imu = (
                    (imu["timestamp_ns"] >= prev_ts) &
                    (imu["timestamp_ns"] <= ts)
                )

                imu_slice = imu[mask_imu]

                if len(imu_slice) >= 2:
                    imu_t = imu_slice["timestamp_ns"].to_numpy() * 1e-9
                    gx = imu_slice["gx"].to_numpy()
                    gy = imu_slice["gy"].to_numpy()
                    gz = imu_slice["gz"].to_numpy()

                    gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)

                    # Approximate integrated rotation angle
                    imu_angle = np.trapz(gyro_norm, imu_t)

                    visual_angles.append(visual_angle)
                    imu_angles.append(imu_angle)
                    times.append((ts - cam["timestamp_ns"].iloc[0]) * 1e-9)

    prev_gray = gray
    prev_kp = kp
    prev_des = des
    prev_ts = ts

visual_angles = np.array(visual_angles)
imu_angles = np.array(imu_angles)
times = np.array(times)

print("Compared frame pairs:", len(times))

if len(times) == 0:
    raise RuntimeError("No valid camera/IMU rotation comparisons.")

corr = np.corrcoef(visual_angles, imu_angles)[0, 1]

print("Visual angle mean:", visual_angles.mean())
print("IMU angle mean:", imu_angles.mean())
print("Correlation:", corr)

plt.figure()
plt.plot(times, visual_angles, label="visual rotation angle")
plt.plot(times, imu_angles, label="IMU integrated gyro angle")
plt.xlabel("time [s]")
plt.ylabel("rotation angle between frames [rad]")
plt.title("Camera vs IMU Rotation Agreement")
plt.legend()
plt.grid(True)
plt.savefig("camera_imu_rotation_sync.png")
plt.show()

print("Saved camera_imu_rotation_sync.png")