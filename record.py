import csv
from datetime import datetime
from pathlib import Path
import depthai as dai
from depthai_sdk import OakCamera, RecordType

OUTPUT_DIR = Path("./recordings").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate a unique filename using the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
imu_csv_path = OUTPUT_DIR / f"imu_{timestamp}.csv"

with open(imu_csv_path, mode="w", newline="") as imu_file:
    imu_writer = csv.writer(imu_file)
    imu_writer.writerow([
        "timestamp_sec", "seq_num",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z"
    ])

    def save_imu_packet(packet):
        accel = packet.acceleroMeter
        gyro = packet.gyroscope
        ts = accel.getTimestampDevice().total_seconds()
        imu_writer.writerow([
            f"{ts:.6f}", accel.sequence,
            accel.x, accel.y, accel.z,
            gyro.x, gyro.y, gyro.z
        ])

    with OakCamera() as oak:
        # 1. Color camera stream (1080p, 30 FPS, hardware H.265 encoded on VPU)
        color = oak.create_camera('color', resolution='1080p', fps=30, encode='H265')
        
        # 2. Left and Right mono stereo pair (400p, 30 FPS, hardware H.265 encoded)
        left = oak.create_camera('left', resolution='400p', fps=30, encode='H265')
        right = oak.create_camera('right', resolution='400p', fps=30, encode='H265')

        # 3. Stereo Depth / Distance stream aligned to RGB color perspective
        stereo = oak.create_stereo(left=left, right=right)
        stereo.config_stereo(align=color)

        # 4. IMU stream (BNO086 at 100 Hz)
        imu = oak.create_imu()
        imu.config_imu(
            sensors=[dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 
            report_rate=100, 
            batch_report_threshold=5
        )

        # 5. Record video and depth streams
        recorder = oak.record(
            outputs=[
                color.out.encoded, 
                left.out.encoded, 
                right.out.encoded, 
                stereo.out.depth
            ], 
            path=str(OUTPUT_DIR), 
            record_type=RecordType.VIDEO
        )

        # 6. Log IMU readings to CSV
        oak.callback(imu.out.main, callback=save_imu_packet)

        # 7. Preview window (press 'Q' while focused on this window to stop and finalize)
        oak.visualize(color.out.camera, scale=0.5, fps=True)

        print(f"Recording video to: {OUTPUT_DIR}")
        print(f"Logging IMU data to: {imu_csv_path}")
        print("Click on the video preview window and press 'Q' to stop recording.")
        
        oak.start(blocking=True)