import os
import cv2
from tqdm import tqdm

ROOT_DIR = "/media/yygx/yygx/Dropbox/Parkinson_Proj/Codes/data/05_16_2025/recording/extracted_videos/undistort_True_colorcorrect_True_devignette_True"

def rotate_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(image_path, rotated)

def rotate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Rotated width
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Rotated height
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_path = video_path + ".rotated_tmp.mp4"

    out = cv2.VideoWriter(tmp_path, codec, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc=f"Rotating video: {os.path.basename(video_path)}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out.write(rotated)

    cap.release()
    out.release()

    os.remove(video_path)
    os.rename(tmp_path, video_path)

def rotate_all(root):
    media_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith((".png", ".mp4")):
                media_files.append(os.path.join(dirpath, filename))

    for path in tqdm(media_files, desc="Rotating all media"):
        if path.endswith(".png"):
            rotate_image(path)
        elif path.endswith(".mp4"):
            rotate_video(path)

if __name__ == "__main__":
    rotate_all(ROOT_DIR)
