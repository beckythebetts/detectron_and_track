from pathlib import Path

# ******* GENERAL *******
DATASET = Path("Datasets") / 'filter_test' / 'filter00.h5'
MASK_RCNN_MODEL = Path("Models") / 'filter_test'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (2048, 2048)
REMOVE_EDGE_CELLS = True

# ******* EPI THRESHOLDING *******
THRESHOLD = 250

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 3
TRACK = True
CLEAN_TRACKS = True
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = 50 # Set as None to view all (slow)

# ******* FEATURE EXTRACTION *******
BATCH_SIZE = 100
PLOT_FEATURES = True
TRACKS_PLOT = False
SHOW_EATING = False


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




