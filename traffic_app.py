import streamlit as st
import cv2
import os
import matplotlib.pyplot as plt

st.title("🚦 AI Traffic Congestion Detector")
uploaded_video = st.file_uploader("Upload CCTV Traffic Footage", type=["mp4"])

if uploaded_video:
    # Save video
    with open("input.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    # Your existing YOLOv5 detection code
    st.write("🔍 Analyzing traffic...")
    !python detect.py --weights yolov5s.pt --source input.mp4 --save-txt
    
    # Your congestion analysis code
    vehicle_counts = []
    labels_dir = "runs/detect/exp/labels"
    for txt_file in sorted(os.listdir(labels_dir)):
        with open(os.path.join(labels_dir, txt_file), "r") as f:
            vehicle_counts.append(sum(1 for line in f if int(line.split()[0]) in [2,3,5,7]))
    
    # Show graph
    fig, ax = plt.subplots()
    ax.plot(vehicle_counts)
    ax.axhline(15, color='red', linestyle='--')
    st.pyplot(fig)
    
    # Alert
    peak = max(vehicle_counts)
    st.error(f"🚨 Peak congestion: {peak} vehicles") if peak >15 else st.success("✅ Traffic normal")
