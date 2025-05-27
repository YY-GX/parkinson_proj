import cv2
import csv
from tqdm import tqdm

# === CONFIGURATION ===
base_path = "/media/yygx/yygx/Dropbox/Parkinson_Proj/Codes/data/05_16_2025/recording/extracted_videos/undistort_True_colorcorrect_True_devignette_True/PD2"
video_path = f'{base_path}/PD2.vrs.mp4'
csv_path = f'{base_path}/05_20_2025_yue_raw.csv'
output_path = f'{base_path}/labeled_output.mp4'

# base_path = "/media/yygx/yygx/Dropbox/Parkinson_Proj/Codes/data/05_16_2025/recording/extracted_videos/undistort_True_colorcorrect_True_devignette_True/PD2-pt2"
# video_path = f'{base_path}/PD2-pt2.vrs.mp4'
# csv_path = f'{base_path}/05_20_2025_yue.csv'
# output_path = f'{base_path}/labeled_output.mp4'

# === LOAD ANNOTATIONS ===
annotations = []
with open(csv_path, newline='') as csvfile:
    first_line = csvfile.readline().strip().replace('"', '')
    fieldnames = first_line.split(';')

    reader = csv.DictReader(csvfile, delimiter=';', fieldnames=fieldnames)
    # reader = csv.DictReader(csvfile, delimiter=',')

    for row in reader:
        try:
            start = float(row['#starttime']) * 1000
            end = float(row['#endtime']) * 1000
            label = row['all_tiers'].strip()
            annotations.append((start, end, label))
        except ValueError:
            continue

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === PROCESS FRAMES WITH PROGRESS BAR ===
frame_num = 0
with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ms = (frame_num / fps) * 1000
        overlapping = [label for start, end, label in annotations if start <= ms <= end]
        label_to_draw = overlapping[-1] if overlapping else ''

        if label_to_draw:
            cv2.putText(frame, label_to_draw, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)
        frame_num += 1
        pbar.update(1)

cap.release()
out.release()

print(f"Labeled video saved to: {output_path}")
