import cv2
from pathlib import Path
from gaze_detector import GazeDetector
from gaze_detector_eval import eval_report, multiclass_report

path3 = Path('/Users/issandungu/Downloads/Columbia Gaze Data Set')

def parse_fname(f_name):
    """File name format - 0001_2m_0P_10v_20H
    Returns: (vertical direction, horizontal direction) as integers"""

    parts = f_name.split("_")
    v_deg = next((int(p[:-1]) for p in parts if p.endswith('V')), 0)
    h_deg = next((int(p[:-1]) for p in parts if p.endswith('H')), 0)

    return v_deg, h_deg # Returns vertical and horizontal positions

def deg_to_labels(v_deg, h_deg, threshold=5):
    """Converts/maps the gaze direction (degrees) to our 5-class integer labels (1 = centre/diagonal, 2=up, 3=down, 4=left, 5=right)"""
    if abs(v_deg) <= threshold and abs(h_deg) <= threshold:
        return 1
    ud_abs = abs(v_deg) # up-down
    lr_abs = abs(h_deg) # left-right

    if v_deg > threshold and ud_abs >= lr_abs:
        return 2
    if v_deg < -threshold and ud_abs >= lr_abs:
        return 3
    if h_deg > threshold:
        return 4
    if h_deg < -threshold:
        return 5
    return 1

def gaze_direction_eval_func():
    detector = GazeDetector()
    y_true, y_pred = [], []

    # Counters
    n_found = 0
    n_no_face = 0
    n_parse_errors = 0
    n_collected = 0

    for img_path in sorted(path3.rglob('*.jpg')):
        n_found += 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm_result = detector.landmark_vals(frame_rgb)

        if not lm_result or len(lm_result.face_landmarks) == 0:
            n_no_face += 1
            continue

        try:
            v_deg, h_deg = parse_fname(img_path.stem)
        except Exception:
            n_parse_errors += 1
            continue

        gt_label = deg_to_labels(v_deg, h_deg)
        pred_label, _ = detector.gaze_direct_detect(lm_result, frame_rgb)

        y_true.append(gt_label)
        y_pred.append(pred_label)
        n_collected += 1

    # Summary -----------------------------------
    print(f"\n--- Summary ---")
    print(f'    Images found: {n_found} ')
    print(f'    No faces detected: {n_no_face}')
    print(f'    Parse errors: {n_parse_errors}')
    print(f'    Collected : {n_collected}')
    # -------------------------------------------------------

    if not y_true:
        print('No predictions collected- check file path.')
        return

    label_names = ['centre', 'up', 'down', 'left', 'right']

    multiclass_report(y_true, y_pred, label_names, label='Gaze Direction - Columbia Gaze (Dataset)')

if __name__ == '__main__':
    gaze_direction_eval_func()


