import cv2
import glob
from pathlib import Path

IMAGE_DIR = "camera_calib_images"

# Change this to your checkerboard inner corners
CHECKERBOARD = (7, 7)

images = sorted(glob.glob(f"{IMAGE_DIR}/*.png") + glob.glob(f"{IMAGE_DIR}/*.jpg"))

valid = 0
invalid = 0

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        valid += 1
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, found)
        cv2.imshow("valid checkerboard", vis)
        cv2.waitKey(80)
    else:
        invalid += 1
        print("No checkerboard:", img_path)

cv2.destroyAllWindows()

print("Total images:", len(images))
print("Valid checkerboard images:", valid)
print("Invalid images:", invalid)