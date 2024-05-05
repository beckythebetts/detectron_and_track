from pathlib import Path
import gc

gc.enable()



DIRECTORY = Path('03')

# VIEW MASKS:
MAX_IMAGES = 50

# TRACKING
OVERLAP_THRESHOLD = 0.5
FRAME_MEMORY = 3
TRACK = True
VIEW_TRACKS = True # Save labelled tracked images

