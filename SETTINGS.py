from pathlib import Path
import gc

gc.enable()



DIRECTORY = Path('03')
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}

# TRACKING
OVERLAP_THRESHOLD = 0.5
FRAME_MEMORY = 3
TRACK = True
VIEW_TRACKS = True # Save labelled tracked images


