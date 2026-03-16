# Suggestion:
# Add on - 1. (Backup) - Callibration stage (Record their baseline angles and then define "Bad Posture" as a percentage deviation (e.g., more than 15% off from baseline))
#          2. (Backup) - Focus Score
#          3. Depth (Z-Axis), mediapipe provide z coordinates for depth, comparing the z of the ear to the shoulder
#          4. Environment detection (Privacy achievable, if people walk outisde then blur out the screen)
#          5. Shoulder callibration (What does it do when the camera is tilted?)
#          6. First replace with the new mediapipe 

import cv2
import time
import winsound # For reminder
import math as m 
import mediapipe as mp
from mediapipe.tasks import python # Newly added, to support 
from mediapipe.tasks.python import vision # Newly added, to support

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
# ------- THE OLD WAY ------
# Used old way to extract key points, which may not be supported in the future.
# The new way requires us to download the file directly
# mp_pose = mp.solutions.pose 
# pose = mp_pose.Pose()

# ------- THE NEW WAY -------
# Path to the new downloaded .task file, 
# An Object-Oriented API designed to deliver same experience across different platform (Android, iOS, etc.)
model_path = r"C:\Users\PC\Documents\Masters\Computer Vision\PosturePro\python-app\pose_landmarker_full.task"

with open(model_path, 'rb') as f:
    model_data = f.read()

# Configure Options
base_options = python.BaseOptions(model_asset_buffer=model_data)
# References: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions 
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,   # Used Video for handling the frame captured through webcam 
)

# Create the 'Detector' (The actual AI engine)
detector = vision.PoseLandmarker.create_from_options(options)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0) # Open the connection to the hardware, 0 means the built-in webcam

    if not cap.isOpened(): # Checking if the connection to the webcam is built
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size) # For saving video

    while True:
        successOrNot, image = cap.read() 
        # read() return if the camera actually successfully grabbed a picture.
        # And the image content
        if not successOrNot:
            print("Skipping empty frame.")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # To fix the color, as Mediapipe read img as RGB, but cv2 read it as BGR    

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        cuurent_times = int(time.time() * 1000) # Get the timestamp which is required for VIDEO mode

        # We store the answer in a new variable called 'results'
        # References: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#video_2 
        results = detector.detect_for_video(mp_image, cuurent_times)

        if not results.pose_landmarks:
            cv2.putText(image, "No pose detected", (10, 30), font, 0.9, red, 2)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

# ==========================================================================================================
# ============================= TO GET THE KEY POINTS OF THE WHOLE BODY STRUCTURE ==========================
# ==========================================================================================================

# References: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker 
# Pose landmarker: 
# 0: nose, 7: left ear, 8: right ear, 11: left shoulder, 12: right shoulder, 23: left hip
        lm = results.pose_landmarks[0] # Get the list of the points for the first person detected.
        
        h, w = image.shape[:2]
        
        # Current mechanism: no shoulder leveling function, and 
        try:
            # lm.landmark[11].x works also, but lm.landmark[lmPose.LEFT_SHOULDER].x prevents accidentally typing.
            l_shldr_x = int(lm[11].x * w) # Multiply them by width or height to get exact pixel value on the img.
            l_shldr_y = int(lm[11].y * h)
            l_shldr_z = lm[11].z # z is not multiplied with width/height as it is a body proportion ratio.
            r_shldr_x = int(lm[12].x * w)
            r_shldr_y = int(lm[12].y * h)
            r_shldr_z = lm[12].z
            l_ear_x = int(lm[7].x * w)
            l_ear_y = int(lm[7].y * h)
            l_hip_x = int(lm[23].x * w)
            l_hip_y = int(lm[23].y * h)
            nose_z = lm[0].z
            
            m_shldr_z = (l_shldr_z + r_shldr_z) / 2 # To get the middle of the body, preventing bias to one side shoulder
        
            # New features 1: To check if user is leaning too close to the screen, "turtle neck" check!
            depth_diff = nose_z - m_shldr_z
            
            # New features 2: To check if user's both shoulder is on the same height.
            shoulder_lvl_difference = abs(l_shldr_y - r_shldr_y) # If both shoulder's y coordinates has a large gap, indicates user is not sitting properly
                  
            # This check how wide a person's shoulder appears in the img and calculate the distance with the screen
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            if offset < 350 and offset > 200:   # User staying too far to the screen
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.7, green, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.7, red, 2)
 
             # The logic of how does a good posture should look like
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y) 
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            angle_text_string = f"Neck : {int(neck_inclination)} Torso : {int(torso_inclination)} NF1: {round(float(depth_diff), 2)} NF2: {int(shoulder_lvl_difference)}"
            suggestions = ""

            depth_threshold = -0.8 # Setting the depth threshold on proper sitting position.
             
            if offset > 350: # 1. Checking user's distance to the screen
                bad_frames += 1
                good_frames = 0
                color = yellow
                suggestions = "Too close, please move back."
            elif offset < 250:
                bad_frames += 1
                good_frames = 0
                color = yellow
                suggestions = "Too far, please move closer."
            elif (depth_diff < depth_threshold): # 2. Checking turtle neck
                good_frames = 0
                bad_frames += 1
                color = red
                suggestions = "You are leaning too forward to the screen! (Turtle neck)"
            elif (shoulder_lvl_difference > 20): # 3. Check if shoulder is leaning one side
                good_frames = 0
                bad_frames += 1
                color = red
                suggestions = "Your shoulder is leaning to one side!"
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
            elif (neck_inclination <= 30 and torso_inclination <= 10 and 
                depth_diff > depth_threshold and 
                shoulder_lvl_difference < 20): # 4. Checking if everything is good.
                bad_frames = 0
                good_frames += 1
                color = light_green
            else:
                good_frames = 0
                bad_frames += 1
                color = red
                suggestions = "You are not sitting straight!"
                        
            # Writes the text directly onto the pixels of the webcam frame.
            cv2.putText(image, angle_text_string, (10, 30), font, 0.7, color, 2)

            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            if bad_time > 60:
                sendWarning()

            # These draw the "skeleton" we see.
            if good_time > 0:
                cv2.putText(image, f'Good Posture Time: {round(good_time, 1)}s', (10, h - 20), font, 0.7, green, 2)
            else:
                cv2.putText(image, f'Bad Posture Time: {round(bad_time, 1)}s Reason: {suggestions}', (10, h - 20), font, 0.7, red, 2)

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