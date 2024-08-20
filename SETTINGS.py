from pathlib import Path
import h5py

# ******* GENERAL *******
DATASET = Path("Datasets") / 'danhighres' / 'dan3.h5'
MASK_RCNN_MODEL = Path("Models") / 'Daniel_highres_fixedsize'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (1002, 1004)
REMOVE_EDGE_CELLS = True

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
BATCH_SIZE = 100
PLOT_FEATURES = False
TRACKS_PLOT = True
SHOW_EATING = True
NUM_FRAMES_EATEN_THRESHOLD = 10


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




