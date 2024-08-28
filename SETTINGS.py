from pathlib import Path
import h5py

# ******* GENERAL *******
DATASET = Path("Datasets") / 'filter_test' / 'no_filter00.h5'
#DATASET = Path("Datasets") / '04_short_testing.h5'
MASK_RCNN_MODEL = Path("Models") / 'Daniel_highres_fixedsize'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (2048, 2048)
REMOVE_EDGE_CELLS = True
NUM_FRAMES = 1200

# ******* EPI THRESHOLDING *******
THRESHOLD = 50

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 3
TRACK = True
CLEAN_TRACKS = True
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = 50 # Set as None to view all (slow)

# ******* FEATURE EXTRACTION *******
BATCH_SIZE = 50
PLOT_FEATURES = False
TRACKS_PLOT = False
SHOW_EATING = False
NUM_FRAMES_EATEN_THRESHOLD = 20


# ******* DIRECTORY STRUCTURE *******
# - 'Models'
#   - model name
#       - 'Training_Data'
#           - 'train'
#               - 'Images' .jpegs
#               - 'labels.json'
#           - 'validate'
#               - 'Images' .jpegs
#               - 'labels.json'




