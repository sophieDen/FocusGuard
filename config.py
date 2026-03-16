# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 15

# Lighting thresholds
# Overall brightness thresholds (0-255 scale)
LIGHTING_DARK_THRESHOLD      = 60    # Below this = too dark overall
LIGHTING_BRIGHT_THRESHOLD    = 200   # Above this = too bright overall

# Histogram-based thresholds
LIGHTING_LOW_INTENSITY_RATIO = 0.7   # If 70%+ pixels are in dark range (0-85), too dark
LIGHTING_HIGH_INTENSITY_RATIO = 0.5  # If 50%+ pixels are in bright range (170-255), too bright
LIGHTING_DARK_PIXEL_THRESHOLD = 85   # Pixels below this = "dark"
LIGHTING_BRIGHT_PIXEL_THRESHOLD = 170 # Pixels above this = "bright"

# Contrast detection (bright screen in dark room)
LIGHTING_CONTRAST_THRESHOLD  = 80    # Difference between center and periphery
LIGHTING_CENTER_RATIO = 0.3          # Central 30% of frame (face/screen area)


# Gaze thresholds
GAZE_DISTRACTION_SECONDS     = 3.0   # seconds looking away before warning
GAZE_DROWSY_EAR_THRESHOLD    = 0.25  # eye aspect ratio below = drowsy
GAZE_DROWSY_SECONDS          = 2.0   # seconds eyes closed before warning

# Posture thresholds
POSTURE_BAD_CONFIDENCE       = 0.75  # model confidence to trigger warning
POSTURE_WARNING_COOLDOWN     = 30    # seconds between repeated warnings

# Alert system 
ALERT_COOLDOWN_SECONDS       = 20    # min gap between same alert repeating