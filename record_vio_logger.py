import argparse
import csv
import struct
import threading
import time
from pathlib import Path

import cv2
import serial


G = 9.80665

ACCEL_LSB_PER_G = 13446.0
GYRO_LSB_PER_DPS = 131.0

PAYLOAD_STRUCT = struct.Struct("<IIhhhhhh")


def xor_checksum(data: bytes) -> int:
    ck = 0
    for b in data:
        ck ^= b
    return ck


def read_packet(ser):
    while True:
        b = ser.read(1)
        if not b:
            return None

        if b == b"\xAA":
            b2 = ser.read(1)
            if b2 == b"\x55":
                break

    payload = ser.read(PAYLOAD_STRUCT.size)
    if len(payload) != PAYLOAD_STRUCT.size:
        return None

    checksum = ser.read(1)
    if len(checksum) != 1:
        return None

    if xor_checksum(payload) != checksum[0]:
        return None

    timestamp_us, packet_id, ax, ay, az, gx, gy, gz = PAYLOAD_STRUCT.unpack(payload)

    ax_mps2 = ax / ACCEL_LSB_PER_G * G
    ay_mps2 = ay / ACCEL_LSB_PER_G * G
    az_mps2 = az / ACCEL_LSB_PER_G * G

    gx_rads = gx / GYRO_LSB_PER_DPS * 3.141592653589793 / 180.0
    gy_rads = gy / GYRO_LSB_PER_DPS * 3.141592653589793 / 180.0
    gz_rads = gz / GYRO_LSB_PER_DPS * 3.141592653589793 / 180.0

    acc_norm = (ax_mps2**2 + ay_mps2**2 + az_mps2**2) ** 0.5
    gyro_norm = (gx_rads**2 + gy_rads**2 + gz_rads**2) ** 0.5

    # Loose sanity filter for handheld motion.
    if acc_norm < 3.0 or acc_norm > 30.0:
        return None

    if gyro_norm > 12.0:
        return None

    return {
        "host_time_ns": time.monotonic_ns(),
        "imu_time_us": timestamp_us,
        "packet_id": packet_id,

        "ax_mps2": ax_mps2,
        "ay_mps2": ay_mps2,
        "az_mps2": az_mps2,

        "gx_rads": gx_rads,
        "gy_rads": gy_rads,
        "gz_rads": gz_rads,

        "raw_ax": ax,
        "raw_ay": ay,
        "raw_az": az,
        "raw_gx": gx,
        "raw_gy": gy,
        "raw_gz": gz,
    }


def imu_thread_fn(ser, rows, stop_event):
    while not stop_event.is_set():
        packet = read_packet(ser)
        if packet is not None:
            rows.append(packet)


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default="vio_recording")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(2.0)
    ser.reset_input_buffer()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    imu_rows = []
    cam_rows = []

    stop_event = threading.Event()
    thread = threading.Thread(
        target=imu_thread_fn,
        args=(ser, imu_rows, stop_event),
        daemon=True,
    )
    thread.start()

    print("Recording camera + IMU...")
    print("Press Q to stop early.")

    start = time.monotonic()
    frame_index = 0

    try:
        while time.monotonic() - start < args.duration:
            ret, frame = cap.read()
            if not ret:
                continue

            host_time_ns = time.monotonic_ns()

            filename = f"{frame_index:06d}.png"
            cv2.imwrite(str(frames_dir / filename), frame)

            cam_rows.append({
                "host_time_ns": host_time_ns,
                "frame_index": frame_index,
                "filename": filename,
            })

            cv2.imshow("camera", frame)

            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        thread.join(timeout=1.0)

        cap.release()
        ser.close()
        cv2.destroyAllWindows()

    write_csv(out_dir / "imu.csv", imu_rows)
    write_csv(out_dir / "cam.csv", cam_rows)

    print("Saved:", out_dir)
    print("IMU rows:", len(imu_rows))
    print("Camera frames:", len(cam_rows))


if __name__ == "__main__":
    main()