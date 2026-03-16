```
├── main.py                  # Entry point — runs everything
├── config.py                # Shared thresholds, camera settings, constants - feel free to change for your module
├── README.md
│
├── core/
│   ├── monitor.py           # Merges all modules
│   ├── base_detector.py     # Abstract base class everyone inherits from
│   └── alert.py             # Unified alert/warning system
│
├── modules/
│   ├── gaze/
│   │   └── gaze_detector.py     # Issa and Brad work here
│   ├── posture/
│   │   └── posture_detector.py  # Jin and Penghui work here
│   └── lighting/
│       └── lighting_detector.py # Sophie work here
│
└── data/
    └── models/              # Any saved ML model files
```
