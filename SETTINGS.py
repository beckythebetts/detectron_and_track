from pathlib import Path
import gc

gc.enable()



DIRECTORY = Path('03')

# VIEW MASKS:
MAX_IMAGES = 50

# TRACKING
THRESHOLD = 0.5
TRACK_CLIP_LENGTH = 3
FRAME_MEMORY = 1
TRACK = True
VIEW_TRACKS = True # Save labelled tracked images

