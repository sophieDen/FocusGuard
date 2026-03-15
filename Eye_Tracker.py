import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
from collections import Counter

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
#   '.tasks' is standard across models, need base options to tell it what model
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
    # print(f"len: {len(lm_result.face_landmarks)}")
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

def seconds_to_frames(num_frames_count, start_time, user_second_amount=10):
    current_time = time.time()
    t_passed = current_time - start_time
    fps = num_frames_count / (t_passed + 1e-8) # ensuring non-zero division

    secs_in_frames = fps * user_second_amount
    return secs_in_frames



def eye_range_vals(lm_result, frame):


    # lm = landmark // le = left eye // re = right eye // u = upper // l = lower
    lm_le_u = 386              #[384, 387] #[386, 263]
    lm_le_l = 374              #[381, 390] #[374, 382]
    lm_re_u = 159              #[161, 158] #[159, 133]
    lm_re_l = 145              #[163, 153] #[145, 153]

    full_landmarks = lm_result.face_landmarks[0]
    # print(full_landmarks)

    # mapping val positions
    lm_le_u = full_landmarks[lm_le_u]
    lm_le_l = full_landmarks[lm_le_l]
    lm_re_u = full_landmarks[lm_re_u]
    lm_re_l = full_landmarks[lm_re_l]


    #=================
    #   EAR = ((p2 - p6) + (p3 - p5)) / (2*(p1 - p4)) - vertical and horizontal ratio of open or shut

    # left eye
    p1 = np.array([full_landmarks[362].x,full_landmarks[362].y])
    print(p1)
    p2 = np.array([full_landmarks[385].x, full_landmarks[385].y])
    p3 = np.array([full_landmarks[387].x, full_landmarks[387].y])
    p4 = np.array([full_landmarks[263].x, full_landmarks[263].y])
    p5 = np.array([full_landmarks[373].x, full_landmarks[373].y])
    p6 = np.array([full_landmarks[380].x, full_landmarks[380].y])
    le_EAR = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))

    # right eye
    rp1 = np.array([full_landmarks[33].x,  full_landmarks[33].y])
    rp2 = np.array([full_landmarks[160].x, full_landmarks[160].y])
    rp3 = np.array([full_landmarks[158].x, full_landmarks[158].y])
    rp4 = np.array([full_landmarks[133].x, full_landmarks[133].y])
    rp5 = np.array([full_landmarks[153].x, full_landmarks[153].y])
    rp6 = np.array([full_landmarks[144].x, full_landmarks[144].y])
    re_EAR = (np.linalg.norm(rp2 - rp6) + np.linalg.norm(rp3 - rp5)) / (2 * np.linalg.norm(rp1 - rp4))


    #================





    # comparing upper and lower vals prioritising y vals
    h, w, _ = frame.shape

    lm_le_range = []
    lm_re_range = []
    # for left eye only
    # print('Left Eye:') # there are 2 vals top and bottom
    # for upper, lower in (lm_le_u, lm_le_l): # now only one val
    #     upper_vals = int(upper.y * h)
    #     lower_vals = int(lower.y * h)
    #     range = lower_vals - upper_vals
    #     lm_le_range.append(range)
    #     print(range) # getting the distance over the eye
    #
    #     # for left eye only
    # print('Right Eye:')  # there are 2 vals top and bottom
    # for upper, lower in (lm_re_u, lm_re_l):  # currently crossing vals
    #     upper_vals = int(upper.y * h)
    #     lower_vals = int(lower.y * h)
    #     range = lower_vals - upper_vals
    #     lm_re_range.append(range)
    #     print(range)  # getting the distance over the eye

    # print('Left Eye:') # there are 2 vals top and bottom
    # # for upper, lower in (lm_le_u, lm_le_l): # now only one val
    # upper_vals = int(lm_le_u.y * h)
    # lower_vals = int(lm_le_l.y * h)
    # range = lower_vals - upper_vals
    # lm_le_range.append(range)
    # print(range) # getting the distance over the eye
    #
    #     # for left eye only
    # print('Right Eye:')  # there are 2 vals top and bottom
    # # for upper, lower in (lm_re_u, lm_re_l):  # currently crossing vals
    # upper_vals = int(lm_re_u.y * h)
    # lower_vals = int(lm_re_l.y * h)
    # range = lower_vals - upper_vals
    # lm_re_range.append(range)
    # print(range)  # getting the distance over the eye

    return le_EAR, re_EAR#lm_le_range, lm_re_range # convert these to EAR values

#===============
# basic sleep detection
#========================

def eye_sleeping_threshold(lm_le_range, lm_re_range, one_sec_n_frames=40, threshold=0.15, s_duration=10): # needs to apply over time
    n_frames = one_sec_n_frames * s_duration

    if np.mean(lm_le_range) <= threshold or np.mean(lm_re_range) <= threshold and len(lm_re_range) > n_frames :
        print(f"le len: {len(lm_le_range)}, re len: {len(lm_re_range)}")
        return True # sleeping
    else:
        print(f"le len: {len(lm_le_range)}, \nre len: {len(lm_re_range)}")
        return False # awake


#=====================
# screen staring/ no blinks
#=====================

def eye_blink_threshold(lm_le_range, lm_re_range, one_sec_n_frames=25, threshold_frames=3, n_threshold=0.15, b_duration=1):
    '''

    :param lm_le_range: left eye range
    :param lm_re_range: right eye range
    :param n_frames: number of frames count is over ( rolling window)
    :param threshold_frames: number of frames that must be under n_threashold
    :param n_threshold: the threashold for what is considered low enough for a blink value
    :param b_duration: toggles the amount of time that blink is over but no need for more than 1 really
    :return:
    '''
    n_frames = one_sec_n_frames * b_duration

    # outputs a list of arrays needed to flatten them for counter to work
    lm_le_range = np.array(lm_le_range).flatten()
    lm_re_range = np.array(lm_re_range).flatten()
    # print(lm_le_range)


    le_threshold = np.sum(lm_le_range[-threshold_frames:] <= n_threshold) #[n_threshold]
    re_threshold = np.sum(lm_re_range[-threshold_frames:] <= n_threshold)

    if le_threshold >= threshold_frames and re_threshold >= threshold_frames and len(lm_re_range) > n_frames-10: # if true blink occurred
        return True # blinked
    else:
        return False # not blinked


def eye_staring_tracker(blink_threshold_func, one_sec_n_frames, staring_duration=10):
    '''

    :param blink_threshold_func: gets True or False from last Func
    :param staring_seconds: amount of time user is staring at the screen in frames
    :param one_sec_n_frames: calculates the amount of frames per second
    :param staring_duration: how many secs spent staring
    :return:
    '''
    n_frames = one_sec_n_frames * staring_duration # gives threshold amount of time staring in frames
    global staring_seconds
    if blink_threshold_func: # if true
        staring_seconds = 0
        return False # not staring
    else:
        staring_seconds +=1
        if staring_seconds > n_frames: # has threshold been reached
            return True # staring
        else:
            return False # not staring



#=====================================================  TODO #########################################################
# gaze tracker                                          TODO #########################################################
#=====================================================  TODO #########################################################
# eye centre = (inner eye + outer eye) /2
# iris centre = centre point iris
# gaze = iris centre - eye centre / np.linalg.norm(iris centre - eye centre)
# ---- looking ----- #
# left = gaze [0] < 0
# right = gaze [0] > 0
# up = gaze [1] < 0
# down = gaze [1] > 0
# centre =


def gaze_direct_detect(lm_result, frame, c_threshold=0.8):

    h, w, _ = frame.shape

    # including landmarks again
    full_landmarks = lm_result.face_landmarks[0]

    # left eye -  make sure to turn these into arrays of x and y or not gonna work
    le_inner = np.array([full_landmarks[133].x * w, full_landmarks[133].y * h])
    le_outer = np.array([full_landmarks[33].x * w, full_landmarks[33].y * h]) #

    le_eye_centre = (le_inner + le_outer) / 2
    le_iris_centre = np.array([full_landmarks[468].x * w, full_landmarks[468].y * h])

    le_gaze = (le_iris_centre - le_eye_centre) / np.linalg.norm(le_iris_centre - le_eye_centre)

    # right eye
    re_inner = np.array([full_landmarks[362].x * w, full_landmarks[362].y * h])
    re_outer = np.array([full_landmarks[263].x * w, full_landmarks[263].y * h]) #

    re_eye_centre = (re_inner + re_outer) / 2
    re_iris_centre = np.array([full_landmarks[473].x * w, full_landmarks[473].y * h])

    re_gaze = (re_iris_centre - re_eye_centre) / np.linalg.norm(re_iris_centre - re_eye_centre)


    # combining gaze

    full_gaze = (le_gaze + re_gaze) / np.linalg.norm(le_gaze + re_gaze)

    # looking direction logic
    # todo inclulde threashold logic to lrud to stop it being overridden
    # reduce to absolute values

    up = full_gaze[1] < 0 - c_threshold
    down = full_gaze[1] > 0 + c_threshold
    right = full_gaze[0] < 0 - c_threshold # left on screen, right irl
    left = full_gaze[0] > 0 + c_threshold # right on screen, left irl

    # means that if the lower left right logic has the higher absolute value then they will be selected instead
    ud_abs = np.abs(full_gaze[1])
    lr_abs = np.abs(full_gaze[0])

    if (not up) and (not down) and (not left) and (not right): # centre
        return 1, full_gaze
    elif up and (ud_abs > lr_abs): # up
        return 2, full_gaze
    elif down and (ud_abs > lr_abs): # down
        return 3, full_gaze
    elif left: # right on screen, left irl
        return 4, full_gaze
    elif right: # left on screen, right irl
        return 5, full_gaze




gaze_down_frames = 0 # global variable - number of frames that user looks down

def gaze_duration_detect(lm_result, frame, one_sec_n_frames, gaze_threshold=5, c_threshold=0.8):
    '''
    Looks at the amount of time the user looks down, and returns an alert if the gaze_threshold is exceeded.
    :param lm_result: mediapipe face landmarking result
    :param frame: current video frame
    :param one_sec_n_frames: the number of frames over 1 second for the camera used
    :param gaze_threshold: the max number of seconds that user can look down before being alerted
    :param c_threshold:

    - need to include time checker - poss global again
    - need to reset each time that it changes over set frames

    :return: warning if the user looks down for too long, or no alert if normal
    '''

    global gaze_down_frames

    direction, full_gaze = gaze_direct_detect(lm_result, frame, c_threshold=0.8)
    threshold_n_frames = one_sec_n_frames * gaze_threshold

    if direction == 3:
        gaze_down_frames += 1
        if gaze_down_frames >= threshold_n_frames:
            print("Looking down for too long. Are you sure you're still focused?")
            return True
    else:
        gaze_down_frames = 0 # resets back to zero if user stops looking down
        return False








#=================
# rolling temporal memory
#=================

def rolling_temporal_memory(lm_le_range, lm_re_range, n_frames=50):
    if len(lm_le_range) > n_frames and len(lm_re_range) > n_frames: # arbitrary number threshold
        del lm_le_range[0] # left first val
        del lm_le_range[1] # left second val
        del lm_re_range[0] # right first val
        del lm_re_range[1] # right second val







#==========================================================
#==========================================================
#   grab the web cam feed
#==========================================================
#==========================================================

# keep outside of loop or camera lags
feed = cv2.VideoCapture(0) # make sure this is 0 for default camera unless there are others

# rolling temporal memory
lm_le_range, lm_re_range = [], []

# fps
num_frames_count = 0
start_time = time.time()

# stare time counter
staring_seconds = 0

while True:


    #==================================================
    # seconds to frames converter - change user_second_amount
    #====================================================
    num_frames_count += 1
    print(f"frames: {num_frames_count}")
    n_sec_frames = seconds_to_frames(num_frames_count=num_frames_count, start_time=start_time, user_second_amount=1)
    one_sec_n_frames = seconds_to_frames(num_frames_count=num_frames_count, start_time=start_time, user_second_amount=1)


    #=================
    # access webcam
    #=================
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
    if len(lm_result.face_landmarks) >= 1:

        # ========================
        # ========================
        #  face landmarks + checks
        # ========================
        # ========================

        # all face landmarks
        full_landmark_mapping(lm_result=lm_result,frame=frame)

        # eye landmarks
        eyes_landmark_mapping(lm_result=lm_result, frame=frame)

        # eye landmark range vals
        lm_le_range_extract, lm_re_range_extract = eye_range_vals(lm_result=lm_result, frame=frame)
        lm_le_range.append(lm_le_range_extract)
        lm_re_range.append(lm_re_range_extract)

        #rolling temporal memory - n_frames to adjust size of memory
        rolling_temporal_memory(lm_le_range=lm_le_range, lm_re_range=lm_re_range, n_frames=n_sec_frames)

        # sleeping check - n_frames changes num secs monitoring sleep
        print("sl_status:",eye_sleeping_threshold(lm_le_range=lm_le_range, lm_re_range=lm_re_range, one_sec_n_frames=one_sec_n_frames-10))

        # blinked check
        print("b_status:",eye_blink_threshold(lm_le_range=lm_le_range, lm_re_range=lm_re_range, one_sec_n_frames=one_sec_n_frames, threshold_frames=1))
        ebt = eye_blink_threshold(lm_le_range=lm_le_range, lm_re_range=lm_re_range, one_sec_n_frames=one_sec_n_frames, threshold_frames=1)

        # staring check - staring duration in secs, increase for longer delay
        print("stare_status:", eye_staring_tracker(blink_threshold_func=ebt, one_sec_n_frames=one_sec_n_frames, staring_duration=5))

        # gaze direction checker
        print("g_direction:",gaze_direct_detect(lm_result=lm_result, frame=frame))

        # gaze duration checker
        print("g_duration: ", gaze_duration_detect(lm_result=lm_result, frame=frame, one_sec_n_frames=one_sec_n_frames, gaze_threshold=5, c_threshold=0.8))




    #=============
    #   show feed
    #===============
    cv2.imshow("Current feed:", frame)

    #=======================
    # lazy value seperation
    #=========================
    print()

    #======================
    #======================
    #   exit window
    #======================
    #======================

    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to exit the window
        break




feed.release() # frees up the camera












