#!/usr/bin/env python3

import cv2
import os
import numpy as np

# Path to the video file
video_path = "../data/input/slides.mov"

# Create a directory to store the extracted frames
output_dir = "../data/output/"
os.makedirs(output_dir, exist_ok=True)

# Load the video
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
ret, prev_frame = cap.read()  # Read the first frame

if not ret:
    print("Failed to read video")
    exit()

# Convert the first frame to grayscale
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Save the first frame as the first slide
frame_filename = f"{output_dir}/slide_0.jpg"
cv2.imwrite(frame_filename, prev_frame)
print(f"Saved first slide: {frame_filename}")

current_frame = 1
saved_frame_count = 1  # Start counting from the second slide
change_threshold = 1000000  # Adjust this threshold based on sensitivity to changes

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of the video

    # Convert the current frame to grayscale for comparison
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)

    # Sum up the difference values (use np.sum)
    diff_sum = np.sum(frame_diff)

    # If the difference exceeds a certain threshold, we consider it a significant change
    if diff_sum > change_threshold:
        frame_filename = f"{output_dir}/slide_{saved_frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)  # Save the frame
        print(f"Saved: {frame_filename}")
        saved_frame_count += 1

    # Update the previous frame to the current one
    prev_frame_gray = current_frame_gray

    current_frame += 1

cap.release()  # Release the video capture object
print("Frame extraction complete.")