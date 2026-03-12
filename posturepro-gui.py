# Suggestion:
# Add on - 1. (Backup) - Callibration stage (Record their baseline angles and then define "Bad Posture" as a percentage deviation (e.g., more than 15% off from baseline))
#          2. Depth (Z-Axis), mediapipe provide z coordinates for depth, comparing the z of the ear to the shoulder
#          3. Environment detection, not sure if achievable, but this would mean no privacy tho
#          4. Shoulder callibration

import cv2
import time
import math as m
import mediapipe as mp

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2): # ======= FRONT FACING ACCURACY =======
    # Suggestion: to incorperate depth feature
    # Which can use the features that Brad suggested
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def sendWarning():
    print("Warning: Bad posture detected for too long!")

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
mp_pose = mp.solutions.pose 
# Using old way to extract key points, which may not be supported in the future.
# The new way requires us to download the file directly
pose = mp_pose.Pose()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) # Open the connection to the hardware, 0 means the built-in webcam

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size) # For saving video

    while True:
        success, image = cap.read()
        if not success:
            print("Skipping empty frame.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image_rgb)

        if not keypoints.pose_landmarks: # ======= PRIVACY FEATURES =======
            # SUGGESTION: Increment 'privacy_counter'. 
            # If 'privacy_counter' > threshold (e.g., 30 frames),
            # set 'image = np.zeros_like(image)' to black out the display.
            cv2.putText(image, "No pose detected", (10, 30), font, 0.9, red, 2)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

# ==========================================================================================================
# ============================= TO GET THE KEY POINTS OF THE WHOLE BODY STRUCTURE ==========================
# ==========================================================================================================

        lm = keypoints.pose_landmarks # Store the big list of all 33 points that MediaPipe found in that specific frame.
        lmPose = mp_pose.PoseLandmark # Like index book, get to achieve: lmPose.LEFT_SHOULDER without needing to rmb index of left shoulder (11)

        h, w = image.shape[:2]
        
        try:
            # lm.landmark[11].x works also, but lm.landmark[lmPose.LEFT_SHOULDER].x prevents accidentally typing.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y) 
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            angle_text_string = f'Neck : {int(neck_inclination)}  Torso : {int(torso_inclination)}'

             # The logic, how does a good posture look like
             # The current accuracy of this part is not so accurate
             # Shoulder leveling is not checked, e.g: If a user is leaning towards right or left...
            if neck_inclination < 40 and torso_inclination < 10: 
                bad_frames = 0
                good_frames += 1
                color = light_green
            else:
                good_frames = 0
                bad_frames += 1
                color = red
                
            # =======  FOCUS SCORE & STAGNATION START =======
            # SUGGESTION: Calculate Focus Score: (sum(posture_history) / len(posture_history)) * 100.
            # SUGGESTION: Track 'shoulder_movement'. Calculate displacement of l_shldr_x/y from previous frame.
            # If movement < epsilon for 1800 frames (1 min), trigger "Switch Posture" advice.

            # Writes the text directly onto the pixels of the webcam frame.
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, color, 2)

            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            if bad_time > 180:
                sendWarning()

            # These draw the "skeleton" we see.
            if good_time > 0:
                cv2.putText(image, f'Good Posture Time: {round(good_time, 1)}s', (10, h - 20), font, 0.9, green, 2)
            else:
                cv2.putText(image, f'Bad Posture Time: {round(bad_time, 1)}s', (10, h - 20), font, 0.9, red, 2)

            # These are the skeleton that we see 
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1) 
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)

        except Exception as e:
            print(f"Error: {e}")

        video_output.write(image)

        cv2.imshow('MediaPipe Pose', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()