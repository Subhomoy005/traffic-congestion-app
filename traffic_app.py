import streamlit as st
import os
import subprocess
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import tempfile

st.title("🚦 AI Traffic Congestion Detector")

# Step 2: Upload Traffic Video
uploaded_video = st.file_uploader("Upload CCTV Traffic Footage", type=["mp4"])

if uploaded_video:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.getbuffer())
        input_video_path = tmp_file.name

    st.write("🔍 Analyzing traffic...")

    # Step 3: Detect Vehicles using YOLOv5
    # Run detection via subprocess (assumes the YOLOv5 repository is in the 'yolov5' folder)
    detect_command = [
        "python", "detect.py",
        "--weights", "yolov5s.pt",
        "--img", "640",
        "--conf", "0.4",
        "--source", input_video_path,
        "--save-txt"
    ]
    # Run the detection command in the yolov5 directory
    result = subprocess.run(detect_command, cwd="yolov5", capture_output=True, text=True)
    st.text(result.stdout)
    if result.returncode != 0:
        st.error("Detection failed!")
    else:
        st.success("Detection complete!")

    # Step 4: Traffic Congestion Logic
    # Count vehicles per frame (classes: 2=car, 3=bike, 5=bus, 7=truck)
    labels_dir = os.path.join("yolov5", "runs", "detect", "exp", "labels")
    vehicle_counts = []
    if os.path.exists(labels_dir):
        for txt_file in sorted(os.listdir(labels_dir)):
            txt_file_path = os.path.join(labels_dir, txt_file)
            with open(txt_file_path, "r") as f:
                vehicles = 0
                for line in f:
                    try:
                        class_id = int(line.split()[0])
                        if class_id in [2, 3, 5, 7]:
                            vehicles += 1
                    except Exception as e:
                        continue
                vehicle_counts.append(vehicles)
    else:
        st.error("Labels directory not found!")

    # Step 5: Alert System & Visualization
    if vehicle_counts:
        # Plot vehicle counts
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(vehicle_counts, color="blue", label="Vehicles per Frame")
        ax.axhline(15, color="red", linestyle="--", label="Congestion Threshold")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Vehicles Detected")
        ax.set_title(f"Traffic Density Analysis (Peak: {max(vehicle_counts)})")
        ax.legend()
        st.pyplot(fig)

        # Alert based on peak vehicles
        peak = max(vehicle_counts)
        if peak > 15:
            st.error(f"🚨 Peak congestion: {peak} vehicles")
        else:
            st.success("✅ Traffic normal")
    else:
        st.warning("No vehicle counts found.")

    # Process video to add congestion alerts (overlay text on frames)
    processed_video_path = "congestion_alert_output.mp4"
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
        
        frame_num = 0
        congestion_threshold = 15
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Overlay congestion alert if vehicle count exceeds threshold
            if frame_num < len(vehicle_counts) and vehicle_counts[frame_num] > congestion_threshold:
                cv2.putText(frame, "CONGESTION ALERT!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            frame_num += 1
        cap.release()
        out.release()
        
        st.video(processed_video_path)

    # Additional Feature Engineering
    # Calculate road area using first frame dimensions
    cap = cv2.VideoCapture(input_video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        road_area = width * height
        cap.release()
    else:
        road_area = 1000  # default value if video can't be read

    # Feature engineering for the last frame
    if vehicle_counts:
        def calculate_speed(vehicles):
            # Placeholder function for speed estimation
            return 0
        features = {
            "vehicle_count": vehicle_counts[-1],
            "avg_speed": calculate_speed([]),
            "time_of_day": datetime.now().hour,
            "road_occupancy": (vehicle_counts[-1] / road_area) * 100
        }
        st.write("Features for Last Frame:", features)
    else:
        st.warning("No features calculated because vehicle counts are missing.")
