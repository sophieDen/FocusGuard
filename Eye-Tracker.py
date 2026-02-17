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

def Web_cam_feed():
    img = cv2.VideoCapture(1)
    if not img.isOpened():
        err = "Could not open feed"
        exit()
        return err
    return img

