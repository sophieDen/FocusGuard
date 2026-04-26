import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# DATASETS
# 1. Drowsiness Dataset (Driver Drowsiness Dataset - Kaggle)
import kagglehub

path1 = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")

# print("Path to dataset files:", path1)

# 2.

# ---------------------------------------

# Reporting Functions
def eval_report(y_true, y_pred, target_names, label=""):
    print(f'\n==== {label} ====')
    print(classification_report(y_true, y_pred, target_names=target_names)) # 'ALERT' in this case refers to alertness
    print('Confusion Matrix (ROWS = TRUE, COLS = PRED):\n')
    print(confusion_matrix(y_true, y_pred))

def multiclass_report(y_true, y_pred, label_names, label=''):
    print(f'\n==== {label} ====')
    print(classification_report(
        y_true, y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0
    ))
    print('Confusion Matrix (ROWS = TRUE, COLS = PRED):\n')
    print(confusion_matrix(y_true, y_pred))



# ---------------------------------------

## DROWSINESS (EAR) Evaluation
import cv2
from gaze_detector import GazeDetector
from pathlib import Path
import glob

fps_fixed = 25
EAR_THRESHOLD = 0.20

# Get ground truth labels:
def get_label(img_path):
    """Returns 1 if DROWSY, and 0 if ALERT - based on folder name"""
    folder = img_path.parent.name
    if folder == 'Drowsy':
        return 1
    elif folder == 'Non Drowsy':
        return 0
    return -1 # skips any unexpected folders encountered

def drowsiness_eval_func():

    # 1. Group by participant
    from collections import defaultdict # to prevent KeyError's due to possible missing keys
    participant_imgs = defaultdict(list)

    for img_path in sorted(Path(path1).rglob('*.png')):
        label = get_label(img_path)
        if label == -1: # does not exist
            continue
        participant_id = img_path.stem[0] # participant label is first character of the image path
        participant_imgs[participant_id].append((img_path, label))

    y_true_all, y_pred_all = [], []

    # Counters ---------
    n_images_found      = sum(len(v) for v in participant_imgs.values())
    n_load_failed       = 0
    n_no_face           = 0
    n_warmup_skipped    = 0
    n_collected         = 0
    # ------------------------------

    for participant_id, items in participant_imgs.items():
        detector = GazeDetector()

        for img_path, gt_label in items:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm_result = detector.landmark_vals(frame_rgb)

            if not lm_result or len(lm_result.face_landmarks) == 0:
                continue

            le, re = detector.eye_range_vals(lm_result, frame_rgb)
            detector.lm_le_range.append(le)
            detector.lm_re_range.append(re)

            predicted = detector.eye_sleeping_threshold(
                detector.lm_le_range, detector.lm_re_range, fps=fps_fixed, s_duration=1) # Using a shorter sleeping duration threshold for evaluation with image dataset instead of videos

            if predicted is None:
                n_warmup_skipped += 1
                continue

            y_true_all.append(gt_label)
            y_pred_all.append(int(predicted))
            n_collected += 1

    # Summary -----------------------------------
    print(f"\n--- Summary ---")
    print(f"  Images found in dataset : {n_images_found}")
    print(f"  Failed to load (cv2)    : {n_load_failed}")
    print(f"  No face detected        : {n_no_face}")
    print(f"  Skipped (buffer warmup) : {n_warmup_skipped}")
    print(f"  Collected for eval      : {n_collected}")
    # ------------------------------------------------------

    if not y_true_all:
        print('No predictions collected - check that MediaPipe is detecting faces.')

    eval_report(y_true_all, y_pred_all, label='Drowsiness Evaluation (EAR Threshold) - DDD (Dataset)', target_names=["ALERT", "DROWSY"])


# if __name__ == '__main__':
#     drowsiness_eval_func()

