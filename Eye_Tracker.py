import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

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


#================================================================================
#================================================================================
#   '.tasks' is standard accross models, need base options to tell it what model
#================================================================================
#================================================================================


base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task") # make sure model is downloaded
options = vision.FaceLandmarkerOptions(base_options=base_options)
face_landmarker = vision.FaceLandmarker.create_from_options(options) # turns it into a usable object


#==========================================================
#==========================================================
#   get the landmarks - must manually show them
#==========================================================
#==========================================================

def landmark_vals(frame):
    if frame is None:
        return None
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    lm_result = face_landmarker.detect(mp_frame)

    return lm_result # 468 landmarks on the face

def full_landmark_mapping(lm_result, frame): # does the full face, just testing
    full_landmarks = lm_result.face_landmarks[0] # only uses the first face that it detects

    h, w, _ = frame.shape


    for landmark in full_landmarks:
        lm_x = int(landmark.x * w)
        lm_y = int(landmark.y * h)
        cv2.circle(frame, (lm_x, lm_y), 1, (0, 255, 0), -1) # img, xy,size,col, thickness/ neg-fill


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
    print(lm_result)


    # ========================
    # ========================
    #   show feed + landmarks
    # ========================
    # ========================
    full_landmark_mapping(lm_result=lm_result,frame=frame)

    cv2.imshow("Current feed:", frame)

    #======================
    #======================
    #   exit window
    #======================
    #======================

    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to exit the window
        break




feed.release() # frees up the camera












