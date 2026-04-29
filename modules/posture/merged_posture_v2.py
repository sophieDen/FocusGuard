import cv2
import time
import math as m 
import mediapipe as mp
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2): 
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def sendWarning():
    print("Warning: Bad posture detected for too long!")
    # winsound.PlaySound("SystemHand", winsound.SND_ALIAS) # For further improvement

good_frames = 0 
bad_frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# For calibrating user's sitting posture
is_calibrated = False
base_depth = 0
base_neck = 0
base_shoulder_lvl = 0
base_offset = 0.0

# Queue for smoothing the noisy Z-axis data
depth_history = []

# Setting up Pose Landmarker
pose_model_path = "C:/Users/PC/Documents/Masters/Computer Vision/Posture/pose_landmarker_full.task"

with open(pose_model_path, 'rb') as f:
    pose_model_data = f.read()

pose_base_options = python.BaseOptions(model_asset_buffer=pose_model_data)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,   
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


# Setting up Object Detector
obj_model_path = "C:/Users/PC/Documents/Masters/Computer Vision/Posture/efficientdet_lite2.tflite" 

obj_options = vision.ObjectDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=obj_model_path),
    running_mode=vision.RunningMode.VIDEO, 
    max_results=5, 
    score_threshold=0.5, 
    category_allowlist=['cell phone', 'chair']
)
obj_detector = vision.ObjectDetector.create_from_options(obj_options)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened(): 
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size) 

    # Track continuous presence/absence for objects
    phone_first_detected_time = None 
    chair_missing_start_time = None 

    while True:
        successOrNot, image = cap.read() 
        if not successOrNot:
            print("Skipping empty frame.")
            break
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        current_times = int(time.time() * 1000) 

        # Process both pose and objects synchronously
        pose_results = pose_detector.detect_for_video(mp_image, current_times)
        obj_results = obj_detector.detect_for_video(mp_image, current_times)

        # Integrate with object detection logic and ui
        phone_detected_this_frame = False 
        chair_detected_this_frame = False

        if obj_results.detections:
            for detection in obj_results.detections:
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                
                cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
                
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score * 100, 1)
                
                if category_name == 'cell phone':
                    phone_detected_this_frame = True
                elif category_name == 'chair':
                    chair_detected_this_frame = True
                
                label = f'{category_name} ({probability}%)'
                cv2.putText(image, label, (start_point[0], start_point[1] - 10), 
                            font, 0.5, (0, 255, 0), 2)
        
        # Object timers & warnings
        if phone_detected_this_frame:
            if phone_first_detected_time is None:
                phone_first_detected_time = time.time()
            else:
                time_on_screen = time.time() - phone_first_detected_time
                if time_on_screen >= 5.0:
                    cv2.putText(image, "Put away the phone!", (50, 80), 
                                font, 1.2, red, 3, cv2.LINE_AA)
        else:
            phone_first_detected_time = None

        if not chair_detected_this_frame:
            if chair_missing_start_time is None:
                chair_missing_start_time = time.time()
            else:
                time_missing = time.time() - chair_missing_start_time
                if time_missing >= 10.0:
                    cv2.putText(image, "Go find a nice chair!", (50, 130), 
                                font, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
        else:
            chair_missing_start_time = None

        # For Pose Detection Logic and UI
        if not pose_results.pose_landmarks:
            cv2.putText(image, "No pose detected", (10, 30), font, 0.9, red, 2)
            cv2.imshow('MediaPipe Integrated App', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        lm = pose_results.pose_landmarks[0] 
        h, w = image.shape[:2]
        
        try:
            l_shldr_x = int(lm[11].x * w) 
            l_shldr_y = int(lm[11].y * h)
            l_shldr_z = lm[11].z 
            r_shldr_x = int(lm[12].x * w)
            r_shldr_y = int(lm[12].y * h)
            r_shldr_z = lm[12].z
            l_ear_x = int(lm[7].x * w)
            l_ear_y = int(lm[7].y * h)
            nose_x = int(lm[0].x * w)
            nose_y = int(lm[0].y * h)
            nose_z = lm[0].z
            
            target_indices = [0, 7, 11, 12]
            current_confidences = [lm[i].visibility for i in target_indices]
            avg_visibility = sum(current_confidences) / len(current_confidences)
            
            cv2.putText(image, f"Model Confidence: {round(avg_visibility, 2)}", 
                        (w - 250, h - 50), font, 0.6, blue, 2)
                        
            m_shldr_z = (l_shldr_z + r_shldr_z) / 2 
        
            # Moving Average Filter for Z-axis noise
            raw_depth_diff = nose_z - m_shldr_z
            depth_history.append(raw_depth_diff)
            if len(depth_history) > 10: # Use only the last 10 frames to prevent bias from one frame.
                depth_history.pop(0)
            
            # Use the smoothed average for all calculations
            depth_diff = sum(depth_history) / len(depth_history)

            shoulder_lvl_difference = abs(l_shldr_y - r_shldr_y) 
            offset = findDistance(lm[11].x, lm[11].y, lm[12].x, lm[12].y)            
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y) 

            suggestions = ""
            z_delta = 0
            color = green 
            
            # For calibration initialization
            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                base_depth = depth_diff
                base_neck = neck_inclination
                base_shoulder_lvl = shoulder_lvl_difference
                base_offset = offset
                is_calibrated = True
            
            if not is_calibrated:
                cv2.putText(image, "SIT STRAIGHT & PRESS 'C' TO CALIBRATE", (50, h//2), font, 0.8, yellow, 2)
                color = yellow
                suggestions = "Waiting for calibration..."
            else:
                z_delta = base_depth - depth_diff
                
                if offset > (base_offset + 0.15): # 1. Sitting posture condition checking: Checking distance
                    good_frames = 0           
                    bad_frames += 1           
                    color = yellow
                    suggestions = "Too close, please move back."
                elif offset < (base_offset - 0.15):
                    good_frames = 0           
                    bad_frames += 1           
                    color = yellow
                    suggestions = "Too far, please move closer."
                elif z_delta > 0.02: # 2. Sitting posture condition checking: Checking turtle neck 
                    good_frames = 0
                    bad_frames += 1
                    color = red
                    suggestions = "Don't lean forward (Turtle neck)!"
                elif abs(shoulder_lvl_difference - base_shoulder_lvl) > 20: # 3. Sitting posture condition checking: Checking uneven shoulder
                    good_frames = 0
                    bad_frames += 1
                    color = red
                    suggestions = "Your shoulders are uneven."
                elif neck_inclination > 35: # 4. Sitting posture condition checking: Checking tif the user's neck is tilting
                    good_frames = 0
                    bad_frames += 1
                    color = red
                    suggestions = "Straighten your neck."
                elif neck_inclination < 15: 
                    good_frames = 0
                    bad_frames += 1
                    color = red
                    suggestions = "Straighten your neck."
                else: # If none of the condition above is matched then outputting as good posture result
                    good_frames += 1
                    bad_frames = 0
                    color = light_green
            
            angle_text_string = f"Neck : {int(neck_inclination)} Depth from nose to body: {round(float(z_delta), 3)} Shoulder Level: {int(shoulder_lvl_difference)}"
            cv2.putText(image, angle_text_string, (10, 30), font, 0.7, color, 2)

            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            if bad_time > 30:
                sendWarning()

            if good_time > 0: # Retain message display as feedback for debugging, which would be removed in the actual application
                cv2.putText(image, f'Good Posture Time: {round(good_time, 1)}s', (10, h - 20), font, 0.7, green, 2)
            else:
                cv2.putText(image, f'Bad Posture Time: {round(bad_time, 1)}s Reason: {suggestions}', (10, h - 20), font, 0.7, red, 2)

            # Display the animation of lines, etc.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1) 
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (nose_x, nose_y), 7, red, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), color, 4)

        except Exception as e:
            print(f"Error: {e}")

        video_output.write(image)

        cv2.imshow('MediaPipe Integrated App', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()