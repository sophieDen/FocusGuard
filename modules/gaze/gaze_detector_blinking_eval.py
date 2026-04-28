import cv2
import numpy as np
from pathlib import Path
from gaze_detector import GazeDetector
from gaze_detector_eval import eval_report

# Dataset
# 2. Eyeblink8 dataset
path2 = Path('/Users/issandungu/Downloads/eyeblink8') # pulls dataset from local device (download Eyeblink8 on local device and amend file path to run)

def load_blink_frames(path, total_frames): # finds blink frames
    """Parse .tag files = second comma-separated field is the blink label (e.g., -1), anything other than -1 = mid-blink"""
    is_blink = np.zeros(total_frames, dtype=bool)

    with open(path) as f: # reads file path
        for line in f:
            line = line.strip()

            # Skip anything that's not a blink
            if not line or line.startswith('#'):
                continue
            parts = line.split(':')

            if len(parts) < 2:
                continue
            try:
                frame_idx = int(parts[0]) # frame index is first element of 'parts'
                blink_label = int(parts[1]) # blink label is the second element of 'parts'
            except ValueError:
                continue # if any errors encountered, skip and move to next file

            if frame_idx < total_frames: # ensures frame indexes match actual video length
                is_blink[frame_idx] = (blink_label != -1) # sets True if label is not -1 (blink), else False (no blink)

    return is_blink # returns blink frames

def blinking_eval_func():
    y_true_all, y_pred_all = [], []

    # Counters ---------
    n_videos = 0
    n_collected = 0
    # ------------------------------

    for video_path in sorted(path2.rglob('*.avi')): # iterates through each video
        tag_file = video_path.parent / (video_path.stem + '.tag') # finds .tag file
        if not tag_file.exists():
            continue # skips if one does not exist

        n_videos += 1 # increments video counter (for summary)
        cap = cv2.VideoCapture(str(video_path)) # feeds video in
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gt_labels = load_blink_frames(tag_file, n_frames)
        detector = GazeDetector()
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok or idx >= n_frames:
                break

            fps = detector.get_fps() # gets frame rate
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converts colour from BGR to RGB
            lm_result = detector.landmark_vals(frame_rgb) # gets landmark values

            if lm_result and len(lm_result.face_landmarks) > 0:
                le, re = detector.eye_range_vals(lm_result, frame_rgb) # gets eye landmarks
                detector.lm_le_range.append(le) # left eye
                detector.lm_re_range.append(re) # right eye

                predicted_blink = detector.eye_blink_threshold(detector.lm_le_range, detector.lm_re_range, fps, threshold_frames=1) # predicts blink/no blink

                y_true_all.append(int(gt_labels[idx])) # ground truth labels
                y_pred_all.append(int(predicted_blink)) # predicted labels


            n_collected += 1 # counts frames collected
            idx += 1

        cap.release()

    # Summary -----------------------------------
    print(f"\n--- Summary ---")
    print(f'    Videos processed: {n_videos} ')
    print(f'    Frames collected: {n_collected}')
    # -------------------------------------------------------

    if not y_true_all:
        print("No predictions collected - check directory path") # sanity print statement, in case
        return

    eval_report(y_true_all, y_pred_all, label="Blink Detection - Eyeblink8 (Dataset)", target_names=['NO BLINK', 'BLINK']) # runs evaluation report (from gaze_detector_eval.py)

# if __name__ == "__main__": # uncomment to run TODO
#     blinking_eval_func()



# ---------------------------------------------

# Confusion Matrix (ROWS = TRUE, COLS = PRED): # for reference when creating CM plots
#
# [[67049   409]
#  [ 1832  1881]]

# Confusion Matrix Plot
import matplotlib.pyplot as plt
import seaborn as sns

cm2 = np.array([[67049, 409], [1832, 1881]])

labels = ['NO BLINK', 'BLINK']

plt.figure()
sns.heatmap(
    cm2,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title('Blinking Detection - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
# plt.savefig('/Users/issandungu/Desktop/MSc Artificial Intelligence/Coursework/SCC.455 - Computer Vision/Model Evaluation - Coursework/Confusion Matrices/confusion_matrix_blinking.png', dpi=300)
# plt.show()






