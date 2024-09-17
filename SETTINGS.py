from pathlib import Path
import h5py

# ******* GENERAL *******
DATASET = Path("Datasets") / 'filter_test' / 'no_filter00.h5'
# DATASET = Path("Datasets") / 'interval_test' / '3sec.hdf5'
# DATASET = Path("Datasets") / '04_short_testing.h5'
# DATASET = Path('Datasets') / 'danhighres' / 'dan10.h5'
MASK_RCNN_MODEL = Path("Models") / 'filter_test'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (2048, 2048)
#IMAGE_SIZE = (1200, 1200)
REMOVE_EDGE_CELLS = True
with h5py.File(DATASET, 'r') as f:
    num_frames = f['Images'].attrs['Number of frames']
#NUM_FRAMES = 49
NUM_FRAMES = num_frames

# ******* EPI THRESHOLDING *******
THRESHOLD = 200

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
MINIMUM_PIXELS_PER_PATHOGEN = 10


# ******* MODEL TRAINING DIRECTORY STRUCTURE *******
# - 'Models'
#   - model name
#       - 'Training_Data'
#           - 'train'
#               - 'Images' .jpegs
#               - 'labels.json'
#           - 'validate'
#               - 'Images' .jpegs
#               - 'labels.json'




