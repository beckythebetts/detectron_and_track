from pathlib import Path

# ******* GENERAL *******
DIRECTORY = Path('03_test')
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (1200, 1200)

# ******* EPI THRESHOLDING *******
THRESHOLD = 6000

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 3
TRACK = True
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = None # Set as None to view all (slow)
TRACKS_PLOT = True

# ******* FEATURE EXTRACTION *******
PLOT_FEATURES = True

# ******* DIRECTORY STRUCTURE ********
# - 'training_dataset'
#       - 'train'
#           - 'images' (jpegs)
#           - 'labels.json'
#       - 'validate'
#           - 'images' (jpegs)
#           - 'labels.json'
# - 'inference_datset'
#       - 'epi' (tiffs)
#       - 'phase' (jpegs)
# *ALL IMAGES 8-BIT*


