import cv2
import numpy as np
import glob
import yaml

IMAGE_DIR = "camera_calib_images"
CHECKERBOARD = (7, 7)
SQUARE_SIZE = 0.025  # change if your square is not 25mm

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    0.0001
)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

images = sorted(
    glob.glob(f"{IMAGE_DIR}/*.png") +
    glob.glob(f"{IMAGE_DIR}/*.jpg") +
    glob.glob(f"{IMAGE_DIR}/*.jpeg")
)

objpoints = []
imgpoints = []
used_files = []
image_size = None

flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

print("Detecting checkerboards...")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]

    # Try newer SB detector first if available
    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

    if not found:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

    if found:
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_files.append(fname)
    else:
        print("No checkerboard:", fname)

print()
print("Detected:", len(objpoints), "valid images")

if len(objpoints) < 15:
    raise RuntimeError("Too few valid images.")


def calibrate_and_errors(objpoints, imgpoints, image_size):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3
    )

    per_image_errors = []

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            K,
            dist
        )

        diff = imgpoints[i].reshape(-1, 2) - projected.reshape(-1, 2)
        rms = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        per_image_errors.append(rms)

    return ret, K, dist, rvecs, tvecs, np.array(per_image_errors)


print()
print("Initial calibration...")

ret, K, dist, rvecs, tvecs, errors = calibrate_and_errors(
    objpoints,
    imgpoints,
    image_size
)

print("Initial RMS:", ret)
print("Per-image error median:", np.median(errors))
print("Per-image error max:", np.max(errors))

# Reject bad images
threshold = max(1.2, np.median(errors) * 2.0)
good_idx = np.where(errors < threshold)[0]

print()
print("Reject threshold:", threshold)
print("Keeping:", len(good_idx), "/", len(errors))

print()
print("Worst images:")
worst = np.argsort(errors)[::-1][:15]
for idx in worst:
    print(f"{errors[idx]:.3f} px | {used_files[idx]}")

objpoints_good = [objpoints[i] for i in good_idx]
imgpoints_good = [imgpoints[i] for i in good_idx]
files_good = [used_files[i] for i in good_idx]

print()
print("Final calibration with filtered images...")

ret, K, dist, rvecs, tvecs, errors2 = calibrate_and_errors(
    objpoints_good,
    imgpoints_good,
    image_size
)

print()
print("FINAL RMS reprojection error:", ret)
print("FINAL per-image median error:", np.median(errors2))
print("FINAL per-image max error:", np.max(errors2))

print()
print("Camera matrix K:")
print(K)

print()
print("Distortion coefficients:")
print(dist.ravel())

calib = {
    "image_width": int(image_size[0]),
    "image_height": int(image_size[1]),
    "camera_matrix": {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    },
    "distortion_model": "plumb_bob",
    "distortion_coefficients": {
        "k1": float(dist.ravel()[0]),
        "k2": float(dist.ravel()[1]),
        "p1": float(dist.ravel()[2]),
        "p2": float(dist.ravel()[3]),
        "k3": 0.0,
    },
    "rms_reprojection_error": float(ret),
    "median_per_image_error": float(np.median(errors2)),
    "max_per_image_error": float(np.max(errors2)),
    "checkerboard": {
        "inner_corners": list(CHECKERBOARD),
        "square_size_m": float(SQUARE_SIZE),
        "valid_images_used": int(len(objpoints_good)),
    }
}

with open("camera_calib_robust.yaml", "w") as f:
    yaml.dump(calib, f, sort_keys=False)

print()
print("Saved camera_calib_robust.yaml")