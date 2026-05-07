import cv2
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent

CAM_DATA = BASE / "euroc_test" / "mav0" / "cam0" / "data.csv"
IMG_DIR = BASE / "euroc_test" / "mav0" / "cam0" / "data"

# Your file seems to be here
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
    CAM_DATA,
    comment="#",
    header=None,
    names=["timestamp_ns", "filename"]
)

orb = cv2.ORB_create(
    nfeatures=2500,
    scaleFactor=1.2,
    nlevels=8,
    fastThreshold=12
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

R_total = np.eye(3)
t_total = np.zeros((3, 1))

trajectory = []
good_pose_count = 0
failed_pose_count = 0

prev_gray = None
prev_kp = None
prev_des = None

for i, row in cam.iterrows():
    img_path = IMG_DIR / row["filename"]
    img = cv2.imread(str(img_path))

    if img is None:
        print("Missing:", img_path)
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

                # Monocular VO has unknown scale.
                # This uses fake unit scale just to visualize trajectory shape.
                scale = 1.0

                t_total = t_total + R_total @ (t * scale)
                R_total = R @ R_total

                good_pose_count += 1
            else:
                failed_pose_count += 1
        else:
            failed_pose_count += 1

        trajectory.append(t_total.flatten())

        if i % 10 == 0:
            vis = cv2.drawMatches(
                prev_gray,
                prev_kp,
                gray,
                kp,
                good[:80],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            cv2.putText(
                vis,
                f"frame={i} matches={len(good)} good_pose={good_pose_count} failed={failed_pose_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("VO pose test", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    prev_gray = gray
    prev_kp = kp
    prev_des = des

cv2.destroyAllWindows()

trajectory = np.array(trajectory)

print("Good pose estimates:", good_pose_count)
print("Failed pose estimates:", failed_pose_count)

if len(trajectory) > 0:
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 2])
    plt.xlabel("x arbitrary scale")
    plt.ylabel("z arbitrary scale")
    plt.title("Monocular Visual Odometry Trajectory - fake scale")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("vo_trajectory.png")
    plt.show()

    print("Saved vo_trajectory.png")