#!/usr/bin/env python3

# Slide Extractor from Video - A tool to automatically detect slide changes in video recordings and save individual slides as images.

# Copyright (C) 2024 - Doug Inman

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import os
import numpy as np
import sys

def extract_slides(video_path):
    # Extract video filename without extension to use as part of output folder name
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a directory to store the extracted frames using the video name
    output_dir = f"{video_name}_extracted_frames"

    # Check if the output directory already exists
    if os.path.exists(output_dir):
        response = input(f"Directory '{output_dir}' already exists. Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        else:
            print(f"Overwriting the contents of '{output_dir}'.")

    # Create or clean the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    ret, prev_frame = cap.read()  # Read the first frame

    if not ret:
        print("Failed to read video")
        return

    # Convert the first frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Save the first frame as the first slide
    frame_filename = os.path.join(output_dir, f"slide_0.jpg")
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
            frame_filename = os.path.join(output_dir, f"slide_{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save the frame
            print(f"Saved: {frame_filename}")
            saved_frame_count += 1

        # Update the previous frame to the current one
        prev_frame_gray = current_frame_gray

        current_frame += 1

    cap.release()  # Release the video capture object
    print(f"Frame extraction complete. Extracted frames are saved in {output_dir}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_slides.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]

    # Ensure the file exists
    if not os.path.isfile(video_file):
        print(f"File not found: {video_file}")
        sys.exit(1)

    # Call the function to extract slides
    extract_slides(video_file)