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


#==========================================================
#==========================================================
#   grab the web cam feed
#==========================================================
#==========================================================


def web_cam_feed():
    feed = cv2.VideoCapture(1)
    if not feed.isOpened():
        print("Could not open feed")
        return None

    ret, frame = feed.read()
    if not ret:
        print("Could not read image")
        return None

    return frame

frame = web_cam_feed()

#==========================================================
#==========================================================
#   get the landmarks - must manually show them
#==========================================================
#==========================================================

def landmark_vals(frame):
    if frame is not None:
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        lm_result = vision.FaceLandmarker.detect(mp_frame)

    return lm_result


lm_result = landmark_vals(frame)


print(lm_result)
















