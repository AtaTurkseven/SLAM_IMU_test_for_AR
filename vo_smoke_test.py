import cv2
import pandas as pd
import numpy as np
from pathlib import Path

DATASET = Path("euroc_test/mav0/cam0")
CSV_PATH = DATASET / "data.csv"
IMG_DIR = DATASET / "data"

cam = pd.read_csv(CSV_PATH, comment="#", header=None, names=["timestamp_ns", "filename"])

orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8,
    fastThreshold=15
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = None
prev_kp = None
prev_des = None

frame_count = 0
good_frame_count = 0

for i, row in cam.iterrows():
    img_path = IMG_DIR / row["filename"]
    img = cv2.imread(str(img_path))

    if img is None:
        print("Missing image:", img_path)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(gray, None)

    if prev_gray is not None and prev_des is not None and des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:100]

        if len(good_matches) > 30:
            good_frame_count += 1

        vis = cv2.drawMatches(
            prev_gray,
            prev_kp,
            gray,
            kp,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.putText(
            vis,
            f"Frame {i} | keypoints: {len(kp)} | matches: {len(good_matches)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("VO smoke test - ORB matches", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    prev_gray = gray
    prev_kp = kp
    prev_des = des

    frame_count += 1

cv2.destroyAllWindows()

print("Frames checked:", frame_count)
print("Frames with enough matches:", good_frame_count)
print("Match success ratio:", good_frame_count / max(frame_count, 1))