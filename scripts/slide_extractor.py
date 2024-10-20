#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
from skimage.metrics import structural_similarity as ssim

# First Pass: Extract slides from presentation video
def extract_slides_from_presentation(video_path, output_dir, frame_diff_threshold=30, min_area=500, similarity_threshold=0.95, stable_duration=2):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        print("Warning: FPS could not be determined. Using a default FPS of 30.")
        fps = 30  # Set a default value if FPS is unavailable
    
    frames_stable_required = int(fps * stable_duration)  # Number of stable frames required for 2 seconds
    print(f"FPS: {fps}, Frames required to consider slide stable: {frames_stable_required}")
    
    # Variables to hold the current and previous frames
    prev_frame = None
    frame_count = 0
    slide_count = 0
    stable_frame_count = 0  # Count of how long a slide has been stable
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir}. {e}")
        return

    def calculate_similarity(frame1, frame2):
        """Calculate structural similarity between two frames."""
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame1, gray_frame2)
        non_zero_count = np.count_nonzero(diff)
        total_pixels = diff.size
        similarity = 1 - (non_zero_count / total_pixels)
        return similarity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if prev_frame is not None:
            # Calculate the similarity between the current frame and the previous frame
            similarity = calculate_similarity(prev_frame, frame)
            
            if similarity > similarity_threshold:
                # If similarity is high, count this frame as stable
                stable_frame_count += 1
            else:
                # Reset stable frame count if the frame is significantly different
                stable_frame_count = 0
                print(f"Frame {frame_count}: Slide changed significantly (Similarity: {similarity:.2f})")

            # If the slide has been stable for the required duration, capture the slide
            if stable_frame_count >= frames_stable_required:
                slide_filename = os.path.join(output_dir, f"slide_{slide_count:03d}.png")
                cv2.imwrite(slide_filename, frame)
                print(f"Captured slide: {slide_filename} (Frame {frame_count}, Similarity: {similarity:.2f})")
                slide_count += 1
                stable_frame_count = 0  # Reset after saving the slide
        
        # Update the previous frame
        prev_frame = frame

    # Release the video capture object
    cap.release()
    print(f"Processing complete. Extracted {slide_count} slides.")

# Second Pass: Refine using histogram and pixel comparisons
def calculate_histogram_similarity(image1_path, image2_path, blur=True):
    """Calculate histogram similarity between two images."""
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        return 0.0

    # Resize images to the same size for comparison
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Apply stronger Gaussian blur to reduce noise and small variations
    if blur:
        image1 = cv2.GaussianBlur(image1, (9, 9), 0)
        image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert images to HSV color space
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Calculate histograms and normalize
    hist1 = cv2.calcHist([hsv_image1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv_image2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compare the histograms using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return similarity

def calculate_ssim_similarity(image1_path, image2_path):
    """Calculate SSIM between two images."""
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        return 0.0

    # Resize images to the same size for comparison
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    # Compute SSIM
    ssim_value, _ = ssim(image1, image2, full=True)

    return ssim_value

def second_pass_refinement(output_dir, hist_similarity_threshold=0.998, ssim_threshold=0.995, passes=3, hist_weight=0.5):
    """
    Second pass to remove similar slides using a combination of histogram and SSIM,
    with multiple refinement passes and dynamic threshold tightening. 
    Files marked for removal will include the pass number.
    """
    # Get all PNG files in the output directory, ignoring files with '_to_be_removed'
    slide_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    
    for pass_num in range(passes):
        print(f"Running pass {pass_num + 1}/{passes}")
        last_kept_slide_path = None
        
        # Use a copy of the slide_files list for safe iteration while removing items
        for slide in slide_files[:]:
            slide_path = os.path.join(output_dir, slide)

            if last_kept_slide_path is not None:
                # Compare the current slide with the last non-removed slide using histogram similarity and SSIM
                hist_similarity = calculate_histogram_similarity(last_kept_slide_path, slide_path)
                ssim_value = calculate_ssim_similarity(last_kept_slide_path, slide_path)

                # Combine histogram and SSIM values using weighted average
                combined_similarity = hist_weight * hist_similarity + (1 - hist_weight) * ssim_value

                # Adjust the thresholds dynamically across passes
                dynamic_hist_threshold = hist_similarity_threshold - (pass_num * 0.0005)
                dynamic_ssim_threshold = ssim_threshold - (pass_num * 0.002)

                # If similarity exceeds both thresholds, mark the slide for removal and add the pass number to the filename
                if hist_similarity >= dynamic_hist_threshold and ssim_value >= dynamic_ssim_threshold:
                    new_slide_path = os.path.join(output_dir, f"{os.path.splitext(slide)[0]}_to_be_removed_pass_{pass_num+1}.png")
                    os.rename(slide_path, new_slide_path)
                    print(f"Pass {pass_num+1}: Marked {slide} as too similar to {os.path.basename(last_kept_slide_path)} "
                          f"(Hist Similarity: {hist_similarity:.3f}, SSIM: {ssim_value:.3f}, Combined: {combined_similarity:.3f})")

                    # Remove the renamed file from the list
                    slide_files.remove(slide)
                else:
                    print(f"Pass {pass_num+1}: {slide} is different enough from {os.path.basename(last_kept_slide_path)} "
                          f"(Hist Similarity: {hist_similarity:.3f}, SSIM: {ssim_value:.3f}, Combined: {combined_similarity:.3f})")
                    last_kept_slide_path = slide_path  # Update only when the slide is kept
            else:
                last_kept_slide_path = slide_path  # Initialize for the first slide

        print(f"Completed pass {pass_num + 1}/{passes}\n")

# Third Pass: Use SSIM (Structural Similarity Index) to refine further, ignoring '_to_be_removed' files
def third_pass_ssim(output_dir, ssim_threshold=0.95):
    # Get all PNG files in the output directory, ignoring files with '_to_be_removed' in the name
    slide_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])
    
    last_kept_slide_path = None
    
    for slide in slide_files[:]:
        slide_path = os.path.join(output_dir, slide)

        # Skip slides already marked "to_be_removed" from any previous step or pass
        if "_to_be_removed" in slide:
            continue

        if last_kept_slide_path is not None:
            # Load images for SSIM comparison
            image1 = cv2.imread(last_kept_slide_path, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(slide_path, cv2.IMREAD_GRAYSCALE)

            if image1 is not None and image2 is not None:
                image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))  # Resize for consistency
                ssim_value, _ = ssim(image1, image2, full=True)

                if ssim_value > ssim_threshold:
                    # If SSIM is too high, rename the current slide to "to_be_removed"
                    new_slide_path = os.path.join(output_dir, f"{os.path.splitext(slide)[0]}_to_be_removed.png")
                    os.rename(slide_path, new_slide_path)
                    print(f"Marked {slide} as too similar to {os.path.basename(last_kept_slide_path)} (SSIM: {ssim_value:.2f})")

                    # Remove the renamed file from the list
                    slide_files.remove(slide)
                else:
                    print(f"{slide} is different enough from {os.path.basename(last_kept_slide_path)} (SSIM: {ssim_value:.2f})")
                    last_kept_slide_path = slide_path  # Update only when the slide is kept
        else:
            last_kept_slide_path = slide_path  # Initialize for the first slide

if __name__ == "__main__":
    # Ensure the script is called with the movie file argument and pass type
    if len(sys.argv) < 3:
        print("Usage: python slide_extractor.py <video_filename> <pass_type>")
        sys.exit(1)

    # Get the video file name and pass type (first, second, third, all) from the command line
    video_filename = sys.argv[1]
    pass_type = sys.argv[2].lower()

    video_file_path = os.path.join(os.getcwd(), video_filename)  # Use relative path to cwd

    # Extract the base name (without extension) from the video file
    base_name = os.path.splitext(video_filename)[0]
    
    # Define the output folder name to include the original file name
    output_folder = os.path.join(os.getcwd(), f"extracted_slides/{base_name}")

    # Run based on the pass_type
    if pass_type == "first":
        extract_slides_from_presentation(video_file_path, output_folder)
    elif pass_type == "second":
        second_pass_refinement(output_folder)
    elif pass_type == "third":
        third_pass_ssim(output_folder)
    elif pass_type == "all":
        extract_slides_from_presentation(video_file_path, output_folder)
        second_pass_refinement(output_folder)
        third_pass_ssim(output_folder)
    else:
        print("Invalid pass type. Use 'first', 'second', 'third', or 'all'.")
        sys.exit(1)