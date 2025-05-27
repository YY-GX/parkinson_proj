#!/usr/bin/env python3

import argparse
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod

# ---------------------------
# Argument Parsing
# ---------------------------
parser = argparse.ArgumentParser(description="Extract and rotate videos from VRS files with optional image corrections.")
parser.add_argument("--undistort", type=bool, default=True, help="Apply lens undistortion to images.")
parser.add_argument("--color_correct", type=bool, default=True, help="Apply color correction to images.")
parser.add_argument("--devignette", type=bool, default=True, help="Apply devignetting to images.")
parser.add_argument("--debug", type=bool, default=False, help="Debug mode: Process only the smallest VRS file.")
args = parser.parse_args()

# ---------------------------
# Configuration
# ---------------------------
VRS_ROOT = "/media/yygx/yygx/Dropbox/Parkinson_Proj/Codes/data/05_16_2025/recording"
OUTPUT_ROOT = os.path.join(
    VRS_ROOT,
    f"extracted_videos/undistort_{args.undistort}_colorcorrect_{args.color_correct}_devignette_{args.devignette}"
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEVIGNETTING_MASK_FOLDER = "/media/yygx/yygx/Dropbox/Parkinson_Proj/Codes/projectaria_tools/devignetting_masks"

# ---------------------------
# Stream ID Resolution Helper
# ---------------------------
def get_available_stream_id(provider):
    possible_labels = ["camera-rgb", "camera-slam-left", "camera-slam-right", "camera-eyetracking"]
    for label in possible_labels:
        try:
            stream_id = provider.get_stream_id_from_label(label)
            if stream_id is not None:
                return stream_id
        except Exception:
            continue
    return None

# ---------------------------
# Processing Function
# ---------------------------
def process_image(provider, stream_id, frame_index, undistort, color_correct, devignette):
    if devignette:
        provider.set_devignetting_mask_folder_path(DEVIGNETTING_MASK_FOLDER)
        provider.set_devignetting(True)
    else:
        provider.set_devignetting(False)

    provider.set_color_correction(color_correct)

    image_data = provider.get_image_data_by_index(stream_id, frame_index)[0]
    image_array = image_data.to_numpy_array()

    if undistort:
        device_calib = provider.get_device_calibration()
        sensor_label = provider.get_label_from_stream_id(stream_id)
        src_calib = device_calib.get_camera_calib(sensor_label)
        dst_calib = calibration.get_linear_camera_calibration(512, 512, 150, sensor_label)
        # if frame_index == 0:
        #     debug_dir = "./debug/"
        #     os.makedirs(debug_dir, exist_ok=True)
        #     # Save raw before undistortion
        #     cv2.imwrite(os.path.join(debug_dir, f"frame_raw.png"), image_array)
        image_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)
        # if frame_index == 0:
        #     cv2.imwrite(os.path.join(debug_dir, f"frame_undistorted.png"), image_array)

    return image_array

# ---------------------------
# Video Extraction and Saving
# ---------------------------
def extract_and_save(provider, vrs_filename, output_dir, undistort, color_correct, devignette):
    os.makedirs(output_dir, exist_ok=True)

    rgb_stream_id = get_available_stream_id(provider)
    if rgb_stream_id is None:
        print(f"No known IMAGE stream found in {vrs_filename}. Skipping.")
        return

    frame_count = provider.get_num_data(rgb_stream_id)

    first_frame = process_image(provider, rgb_stream_id, 0, undistort, color_correct, devignette)
    height, width = first_frame.shape[:2]
    rotated_size = (height, width)  # 90 deg rotation swaps width and height
    fps = 30  # Estimated FPS

    video_path = os.path.join(output_dir, f"{os.path.basename(vrs_filename)}.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, rotated_size, isColor=True)

    thumbnails = []
    indices = [0, frame_count // 2, frame_count - 1]

    for idx in tqdm(range(frame_count), desc=f"Processing {os.path.basename(vrs_filename)}"):
        frame_array = process_image(provider, rgb_stream_id, idx, undistort, color_correct, devignette)

        # Convert RGB to BGR if needed
        if len(frame_array.shape) == 2:
            frame_to_write = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
        else:
            frame_to_write = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

        # Rotate 90 degrees clockwise before saving
        frame_to_write = cv2.rotate(frame_to_write, cv2.ROTATE_90_CLOCKWISE)

        video_writer.write(frame_to_write)

        if idx in indices:
            thumb_path = os.path.join(output_dir, f"thumbnail_{indices.index(idx) + 1}.png")
            cv2.imwrite(thumb_path, frame_to_write)
            thumbnails.append(thumb_path)

    video_writer.release()

    metadata = {
        "video_path": video_path,
        "frame_count": frame_count,
        "fps": fps,
        "resolution": {"width": rotated_size[0], "height": rotated_size[1]},
        "thumbnails": thumbnails,
        "undistort": undistort,
        "color_correct": color_correct,
        "devignette": devignette
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# ---------------------------
# File Discovery Helper
# ---------------------------
def find_smallest_vrs(root_dir):
    debug_file = "/mnt/ssd1/Dropbox/Parkinson_Proj/Codes/data/05_16_2025/recording/Try.vrs"
    return debug_file
    vrs_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith(".vrs")]
    if not vrs_files:
        return None
    return min(vrs_files, key=lambda x: os.path.getsize(x))

# ---------------------------
# Main Execution Logic
# ---------------------------
if args.debug:
    smallest_vrs = find_smallest_vrs(VRS_ROOT)
    if smallest_vrs:
        print(f"[DEBUG] Processing smallest VRS file: {smallest_vrs}")
        provider = data_provider.create_vrs_data_provider(smallest_vrs)
        if provider:
            session_output_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(os.path.basename(smallest_vrs))[0])
            extract_and_save(provider, smallest_vrs, session_output_dir, args.undistort, args.color_correct, args.devignette)
        else:
            print(f"[DEBUG] Failed to load provider for: {smallest_vrs}")
    else:
        print("[DEBUG] No VRS files found.")
else:
    for root, _, files in os.walk(VRS_ROOT):
        for file in files:
            if file.endswith(".vrs"):
                vrs_path = os.path.join(root, file)
                provider = data_provider.create_vrs_data_provider(vrs_path)
                if provider is None:
                    print(f"Skipping invalid VRS: {vrs_path}")
                    continue
                session_output_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(file)[0])
                extract_and_save(provider, vrs_path, session_output_dir, args.undistort, args.color_correct, args.devignette)
