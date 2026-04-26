import cv2
import numpy as np
from pathlib import Path
from gaze_detector import GazeDetector
from gaze_detector_eval import eval_report

# Dataset
# 2. Eyeblink8 dataset
path2 = Path('/Users/issandungu/Downloads/eyeblink8')

def load_blink_frames(path, total_frames):
    """Parse .tag files = second comma-separated field is the blink label (e.g., -1), anything other than -1 = mid-blink"""
    is_blink = np.zeros(total_frames, dtype=bool)

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip anything that's not a blink
            if not line or line.startswith('#'):
                continue
            parts = line.split(':')

            if len(parts) < 2:
                continue
            try:
                frame_idx = int(parts[0])
                blink_label = int(parts[1])
            except ValueError:
                continue

            if frame_idx < total_frames:
                is_blink[frame_idx] = (blink_label != -1)

    return is_blink

def blinking_eval_func():
    y_true_all, y_pred_all = [], []
    blink_rate_rows = []

    # Counters ---------
    n_videos = 0
    n_collected = 0
    # ------------------------------

    for video_path in sorted(path2.rglob('*.avi')):
        tag_file = video_path.parent / (video_path.stem + '.tag')
        if not tag_file.exists():
            continue

        n_videos += 1
        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        gt_labels = load_blink_frames(tag_file, n_frames)
        detector = GazeDetector()
        idx = 0

        y_pred_video = []

        while True:
            ok, frame = cap.read()
            if not ok or idx >= n_frames:
                break

            fps = detector.get_fps()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm_result = detector.landmark_vals(frame_rgb)

            if lm_result and len(lm_result.face_landmarks) > 0:
                le, re = detector.eye_range_vals(lm_result, frame_rgb)
                detector.lm_le_range.append(le)
                detector.lm_re_range.append(re)

                predicted_blink = detector.eye_blink_threshold(detector.lm_le_range, detector.lm_re_range, fps, threshold_frames=1)

                y_true_all.append(int(gt_labels[idx]))
                y_pred_all.append(int(predicted_blink))
                y_pred_video.append(int(predicted_blink))

            n_collected += 1
            idx += 1

        cap.release()

    # Summary -----------------------------------
    print(f"\n--- Summary ---")
    print(f'    Videos processed: {n_videos} ')
    print(f'    Frames collected: {n_collected}')
    # -------------------------------------------------------

    if not y_true_all:
        print("No predictions collected - check directory path")
        return

    eval_report(y_true_all, y_pred_all, label="Blink Detection - Eyeblink8 (Dataset)", target_names=['NO BLINK', 'BLINK'])

if __name__ == "__main__":
    blinking_eval_func()








