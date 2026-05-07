import cv2
import yaml
import glob
import numpy as np

CALIB_FILE = "camera_calib_robust.yaml"
IMAGE_DIR = "camera_calib_images"

with open(CALIB_FILE, "r") as f:
    calib = yaml.safe_load(f)

w = calib["image_width"]
h = calib["image_height"]

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

images = sorted(glob.glob(f"{IMAGE_DIR}/*.png"))

new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

for path in images[:20]:
    img = cv2.imread(path)
    if img is None:
        continue

    undistorted = cv2.undistort(img, K, dist, None, new_K)

    combined = np.hstack([
        cv2.resize(img, (w, h)),
        cv2.resize(undistorted, (w, h))
    ])

    cv2.imshow("left: original | right: undistorted", combined)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()