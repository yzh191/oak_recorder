import csv
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
RECORDING_FOLDER = Path(r"C:\oak_recorder\recordings\1-14442C10C1D1EFD000")

COLOR_VIDEO_PATH = RECORDING_FOLDER / "CAM_A_bitstream.mp4"

# Automatically find the recorded depth file in the folder
depth_candidates = [
    f for f in RECORDING_FOLDER.iterdir()
    if "depth" in f.name.lower() or "stereo" in f.name.lower()
]
if not depth_candidates:
    files_found = [f.name for f in RECORDING_FOLDER.iterdir()]
    raise FileNotFoundError(
        f"No depth file found in {RECORDING_FOLDER}. Files present: {files_found}"
    )
DEPTH_VIDEO_PATH = depth_candidates[0]
print(f"Using color file: {COLOR_VIDEO_PATH.name}")
print(f"Using depth file: {DEPTH_VIDEO_PATH.name}")

OUTPUT_VIDEO_PATH = RECORDING_FOLDER / "annotated_people.mp4"
OUTPUT_CSV_PATH = RECORDING_FOLDER / "people_detections.csv"

# Load YOLOv8 nano model (automatically downloads weights on first run)
model = YOLO("yolov8n.pt")


def get_distance_meters(depth_frame: np.ndarray, box: tuple) -> float | None:
    """
    Calculates the median distance in meters within the center region of a bounding box
    to avoid background bleed around the edges of a person.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_frame.shape[:2]

    # Clamp coordinates to frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Crop to the inner 40% of the bounding box (torso area) for cleaner depth readings
    box_w = x2 - x1
    box_h = y2 - y1
    cx1 = x1 + int(box_w * 0.3)
    cx2 = x2 - int(box_w * 0.3)
    cy1 = y1 + int(box_h * 0.3)
    cy2 = y2 - int(box_h * 0.3)

    roi = depth_frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None

    # If depth was saved as an 8-bit 3-channel video, convert to single channel
    if roi.ndim == 3:
        roi = roi[:, :, 0]

    # Filter out 0 (invalid/no depth data)
    valid_depths = roi[roi > 0]
    if valid_depths.size == 0:
        return None

    median_val = float(np.median(valid_depths))

    # NOTE ON UNITS:
    # Raw DepthAI depth maps are in millimeters (16-bit). 
    # If your depth video was saved as 16-bit, divide by 1000.0 to get meters.
    # If oak.record() compressed it to 8-bit (0-255), verify your max depth scaling.
    if depth_frame.dtype == np.uint16:
        return round(median_val / 1000.0, 2)
    else:
        # Standard DepthAI 8-bit disparity/depth visualization fallback (adjust scale if needed)
        return round(median_val / 1000.0, 2)


def main():
    color_cap = cv2.VideoCapture(str(COLOR_VIDEO_PATH))
    # CAP_PROP_CONVERT_RGB=0 preserves 16-bit depth values if stored in the container
    depth_cap = cv2.VideoCapture(str(DEPTH_VIDEO_PATH))
    depth_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if not color_cap.isOpened() or not depth_cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video streams in {RECORDING_FOLDER}. "
            "Check exact filenames and extensions (.mp4 / .h265 / .avi)."
        )

    fps = color_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    with open(OUTPUT_CSV_PATH, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_idx", "timestamp_sec", "track_id",
            "confidence", "x1", "y1", "x2", "y2", "distance_m"
        ])

        frame_idx = 0
        while True:
            ret_c, color_frame = color_cap.read()
            ret_d, depth_frame = depth_cap.read()

            if not ret_c or not ret_d:
                break

            timestamp_sec = round(frame_idx / fps, 3)

            # Ensure depth frame matches color resolution (1080p)
            if depth_frame.shape[:2] != (height, width):
                depth_frame = cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # Run YOLOv8 tracking filtered strictly to class 0 ('person')
            results = model.track(color_frame, classes=[0], persist=True, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    dist_m = get_distance_meters(depth_frame, (x1, y1, x2, y2))
                    dist_text = f"{dist_m:.2f}m" if dist_m is not None else "N/A"

                    # Log to CSV
                    writer.writerow([
                        frame_idx, timestamp_sec, track_id,
                        round(conf, 3), int(x1), int(y1), int(x2), int(y2),
                        dist_m if dist_m is not None else ""
                    ])

                    # Draw bounding box & distance label on video
                    label = f"Person #{track_id} | {dist_text}"
                    cv2.rectangle(color_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        color_frame, label, (int(x1), max(20, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

            out_video.write(color_frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

    color_cap.release()
    depth_cap.release()
    out_video.release()
    print(f"Done! Saved video to {OUTPUT_VIDEO_PATH} and CSV to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()