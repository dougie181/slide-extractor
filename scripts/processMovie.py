#!/usr/bin/env python3

import cv2
import os

# Path to the video file
video_path = "../data/input/slides.mov"

# Create a directory to store the extracted frames
output_dir = "../data/output/"
os.makedirs(output_dir, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate

current_frame = 0
frame_interval = int(frame_rate * 3)  # Adjust the interval for each slide (e.g., every 3 seconds)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame if it's the right interval (you can tweak this)
    if current_frame % frame_interval == 0:
        frame_filename = f"{output_dir}/frame_{current_frame}.jpg"
        cv2.imwrite(frame_filename, frame)
    
    current_frame += 1

cap.release()
print("Extraction complete.")