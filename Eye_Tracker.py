import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# use tasks not solutions

# use a mediapipe graph
# face detection model into face landmark model to iris detection
# face landmark subgraph from face landmark module
# iris landmark graph from the iris landmark module

# rendering is done using an iris-and-depth renderer subgraph
# The face landmark subgraph internally uses a face detection subgraph from the face detection module.

# running or cpu and gpu seem to have different implementation

# mediapipe.graphs.iris_tracking.calculators.subgraph
# mp.tasks.vision.face_landmarker.FaceLandmarker.detect()

#
#================================================================================
#================================================================================
#   '.tasks' is standard accross models, need base options to tell it what model
#================================================================================
#================================================================================


base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task") # make sure model is downloaded
options = vision.FaceLandmarkerOptions(base_options=base_options)
face_landmarker = vision.FaceLandmarker.create_from_options(options) # turns it into a usable object


#=============================================================================
#=============================================================================
#   get the landmarks - must manually show them - just gives position vals
#=============================================================================
#=============================================================================

def landmark_vals(frame):
    if frame is None:
        return None
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    lm_result = face_landmarker.detect(mp_frame)

    return lm_result # 468 landmarks on the face




def full_landmark_mapping(lm_result, frame): # does the full face, just testing
    full_landmarks = lm_result.face_landmarks[0] # only uses the first face that it detects

    h, w, _ = frame.shape # used for scaling, '_' ---- dont need channel nums

    for landmark in full_landmarks:
        lm_x = int(landmark.x * w) # mp vals are normalised need scaling up
        lm_y = int(landmark.y * h)
        cv2.circle(frame, (lm_x, lm_y), 1, (0, 255, 0), -1) # img, xy,size,col (bgr), thickness/ neg-fill #



def eyes_landmark_mapping(lm_result, frame): # found the eye idx vals
    lm_le= [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]  # Left eye landmarks

    lm_re = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # Right eye landmarks

    lm_eyes = lm_le + lm_re # both eyes

    full_landmarks = lm_result.face_landmarks[0] # only uses the first face that it detects

    h, w, _ = frame.shape # used for scaling, '_' ---- dont need channel nums

    #================
    # generates the landmarks
    #=======================

    for idx in lm_eyes:
        lm = full_landmarks[idx]
        lm_x = int(lm.x * w) # mp vals are normalised need scaling up
        lm_y = int(lm.y * h)
        cv2.circle(frame, (lm_x, lm_y), 1, (0, 0, 255), -1) # img, xy,size,col (bgr), thickness/ neg-fill #



#=========================================================================
#=========================================================================
#       calculating the amount of time with eyes open/closed
#=========================================================================
#=========================================================================

def eye_staring(lm_result, frame):
    # lm = landmark // le = left eye // re = right eye // u = upper // l = lower
    lm_le_u = [386, 263]
    lm_le_l = [374, 382]
    lm_re_u = [159, 133]
    lm_re_l = [145, 153]

    full_landmarks = np.array(lm_result.face_landmarks[0])

    # mapping val positions
    lm_le_u = full_landmarks[lm_le_u]
    lm_le_l = full_landmarks[lm_le_l]
    lm_re_u = full_landmarks[lm_re_u]
    lm_re_l = full_landmarks[lm_re_l]

    # comparing upper and lower vals prioritising y vals
    h, w, _ = frame.shape

    # for left eye only
    for upper, lower in (lm_le_u, lm_le_l):
        upper_vals = int(upper.y * h)
        lower_vals = int(lower.y * h)

        print(upper_vals - lower_vals) # getting the distance over the eye




def eyes_closed():
    pass









#==========================================================
#==========================================================
#   grab the web cam feed
#==========================================================
#==========================================================

# keep outside of loop or camera lags
feed = cv2.VideoCapture(0) # make sure this is 0 for default camera unless there are others
while True:

    if not feed.isOpened():
        print("Could not open feed")
        exit()

    ret, frame = feed.read()
    if not ret:
        print("Could not read image")
        exit()



    # shows the position of the values
    lm_result = landmark_vals(frame)
    # print(lm_result) #######################################################################################


    # ========================
    # ========================
    #   show feed + face landmarks
    # ========================
    # ========================
    full_landmark_mapping(lm_result=lm_result,frame=frame)

    eyes_landmark_mapping(lm_result=lm_result, frame=frame)

    eye_staring(lm_result=lm_result, frame=frame)

    cv2.imshow("Current feed:", frame)

    #======================
    #======================
    #   exit window
    #======================
    #======================

    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to exit the window
        break




feed.release() # frees up the camera












