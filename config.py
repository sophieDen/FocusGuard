# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 15

# Lighting thresholds
LIGHTING_DARK_THRESHOLD      = 60    # 0-255, below = too dark
LIGHTING_BRIGHT_THRESHOLD    = 200   # 0-255, above = too bright
LIGHTING_CONTRAST_THRESHOLD  = 50    # face vs ambient brightness gap

# Gaze thresholds
GAZE_DISTRACTION_SECONDS     = 3.0   # seconds looking away before warning
GAZE_DROWSY_EAR_THRESHOLD    = 0.25  # eye aspect ratio below = drowsy
GAZE_DROWSY_SECONDS          = 2.0   # seconds eyes closed before warning

# Posture thresholds
POSTURE_BAD_CONFIDENCE       = 0.75  # model confidence to trigger warning
POSTURE_WARNING_COOLDOWN     = 30    # seconds between repeated warnings

# Alert system 
ALERT_COOLDOWN_SECONDS       = 20    # min gap between same alert repeating