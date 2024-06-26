from pathlib import Path

# ******* GENERAL *******
DIRECTORY = Path("04")
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (1200, 1200)
REMOVE_EDGE_CELLS = True

# ******* EPI THRESHOLDING *******
THRESHOLD = 17469

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 3
TRACK = False
CLEAN_TRACKS = True
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = 50 # Set as None to view all (slow)


# ******* FEATURE EXTRACTION *******
BATCH_SIZE = 100
PLOT_FEATURES = True
TRACKS_PLOT = False
SHOW_EATING = False

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


