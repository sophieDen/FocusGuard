import kagglehub
import cv2
from pathlib import Path
from gaze_detector import GazeDetector
from gaze_detector_eval import multiclass_report

# Dataset
# 3. MPIIFaceGaze dataset
path3 = Path(kagglehub.dataset_download("vimal704/mpiifacegaze")) # pulls dataset from Kaggle

data_dir = path3 / "MPIIFaceGaze_preprocessed" # data directory path
img_dir = data_dir / "Image" # img directory path
label_dir = data_dir / "Label" # label directory path


def parse_3d_gaze(gaze_str):
    """Parses '3DGaze' column - contains comma-separated x,y,z unit vectors"""
    x, y, z = map(float, gaze_str.split(','))
    return x, y, z

def gaze_vector_to_label(x, y, threshold=0.10, y_offset=0.12):
    """Maps 3D gaze vector to our 5-class label
    x-value: negative = right, positive = left
    y-value: negative = up, positive = down
    threshold: minimum magnitude needed to classify as up,down,left,right. Else = centre
    y_offset  - accounts for natural downward gaze when looking at laptop, for example

    adding y_offset ---> shifts centre value down to match evaluation dataset properties (i.e., using laptop, looking slightly down...)
    """
    y_adjusted = y - y_offset

    if abs(x) <= threshold and abs(y_adjusted) <= threshold:
        return 1 # centre
    ud_abs = abs(y_adjusted)
    lr_abs = abs(x)

    if y_adjusted > threshold and ud_abs >= lr_abs:
        return 2 # up
    if y_adjusted < -threshold and ud_abs >= lr_abs:
        return 3 # down
    if x > threshold:
        return 4 # left
    if x < -threshold:
        return 5 # right
    return 1 # if diagonal --> becomes centre

def load_labels(label_path):
    """Returns dictionary with the image name (key) and the gaze direction label (value)"""
    labels = {}

    with open(label_path) as f:
        header = f.readline() # to skip header line
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: # likely not a valid image file if the length of this is < 6
                continue

            img_name = Path(parts[0].replace('\\', '/')).name # gets image file name
            gaze_str = parts[5]

            try:
                x, y, z = parse_3d_gaze(gaze_str) # gets x, y and z coordinates from the gaze string
                labels[img_name] = gaze_vector_to_label(x, y) # creates dictionary with image name: label pair -  {img_name: label}
            except Exception:
                continue # moves to next file if any errors thrown
    return labels # returns clean labels

def gaze_direction_eval_func():
    detector = GazeDetector()
    y_true, y_pred = [], []

    # Counters (for summary)
    n_found = 0
    n_no_face = 0
    n_no_label = 0
    n_collected = 0

    for participant in sorted(img_dir.iterdir()): # iterates through each participant image directory
        if not participant.is_dir():
            continue

        label_file = label_dir / (participant.name + '.label') # creates label file name
        if not label_file.exists():
            continue # if doesn't exist, skipped

        label_map = load_labels(label_file) # gets labels
        face_dir = participant / 'face' # creates face directory path

        print(f'Processing {participant.name} ({len(label_map)} labels) ...')

        for img_path in sorted(face_dir.glob('*.jpg')): # loops through each image in the face directory
            n_found += 1 # increments counter

            gt_label = label_map.get(img_path.name) # obtains GT label
            if gt_label is None: # if no GT label found...
                n_no_label += 1 # increment 'no label' counter
                continue # move to next image

            frame = cv2.imread(str(img_path)) # reads image
            if frame is None: # if file is corrupted/not an image, etc...
                continue # moves to next image

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converts colors from BGR to RBG
            lm_result = detector.landmark_vals(frame_rgb) # gets face landmark values

            if not lm_result or len(lm_result.face_landmarks) == 0: # checks if face detected by module
                n_no_face += 1 # increments counter
                continue # skips image and moves to next one

            pred_label, _ = detector.gaze_direct_detect(lm_result, frame_rgb) # gets predicted label

            y_true.append(gt_label) #appends to GT labels
            y_pred.append(pred_label) # appends to predicted labels
            n_collected += 1 # increments 'number of images collected/processed' counter

    # Summary -----------------------------------
    print(f"\n--- Summary ---")
    print(f'    Images found: {n_found} ')
    print(f'    No face detected: {n_no_face}')
    print(f'    No label match: {n_no_label} ')
    print(f'    Images collected/processed: {n_collected} ')
    # -------------------------------------------------------

    if not y_true:
        print('No predictions collected - debug/check paths') # sanity print statement
        return

    label_names = ['centre', 'up', 'down', 'left', 'right']

    multiclass_report(y_true, y_pred, label_names, label='Gaze Direction MPIIFaceGaze (Dataset)')


if __name__ == '__main__':
    gaze_direction_eval_func()

# ---------------------------------------------


# Confusion Matrix Plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm3 = np.array([[0,11325,12,97,98], # hardcoded CM for plotting only
 [0,6887,52,168,259],
 [0,1408,2,27,16],
 [0,9873,16,2382,10],
 [0,9401,14,8,2745]])

labels = ['CENTER', 'UP', 'DOWN', 'LEFT', 'RIGHT']

plt.figure()
sns.heatmap(
    cm3,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title('Gaze Direction Detection - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
# plt.savefig('/Users/issandungu/Desktop/MSc Artificial Intelligence/Coursework/SCC.455 - Computer Vision/Model Evaluation - Coursework/Confusion Matrices/confusion_matrix_gazedirection_mpiifacegaze.png', dpi=300)
# plt.show()


