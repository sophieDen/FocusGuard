# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15

# Lighting thresholds
# Overall brightness thresholds (0-255 scale)
LIGHTING_DARK_THRESHOLD = 60
LIGHTING_BRIGHT_THRESHOLD = 200

# Contrast detection
LIGHTING_CENTER_RATIO = 0.3 # Central 30% of frame

# Gaze thresholds
GAZE_DISTRACTION_SECONDS = 3.0   # seconds looking away before warning
GAZE_DROWSY_EAR_THRESHOLD = 0.25  # eye aspect ratio below = drowsy
GAZE_DROWSY_SECONDS = 2.0   # seconds eyes closed before warning