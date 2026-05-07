import cv2
from pathlib import Path

OUT = Path("camera_calib_images")
OUT.mkdir(exist_ok=True)

CAMERA_INDEX = 1
WIDTH = 640
HEIGHT = 480

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

idx = 40

print("Press SPACE to save image.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("calibration capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = OUT / f"calib_{idx:03d}.png"
        cv2.imwrite(str(filename), frame)
        print("Saved:", filename)
        idx += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()