from pathlib import Path

# ******* GENERAL *******
DATASET = Path("Datasets") / '04_short.h5'
MASK_RCNN_MODEL = Path("Models") / '00'
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (1200, 1200)
REMOVE_EDGE_CELLS = True

# ******* EPI THRESHOLDING *******
THRESHOLD = 17469

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
#               - 'lables.json'
#           - 'validate'
#               - 'Images' .jpegs
#               - 'labels.json'
# - 'Datasets'
#   - dataset name
#       - 'Images'
#           - 'Phase' .jpegs
#           - 'Epi' .TIFs



