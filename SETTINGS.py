from pathlib import Path

# ******* GENERAL *******
DIRECTORY = Path("/home/ubuntu/Documents/det_proj/detectron_and_track/kfold_18_old/train6/test('00',)")
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}
IMAGE_SIZE = (1200, 1200)
IMAGE_SIZE = (822, 920)
REMOVE_EDGE_CELLS = True

# ******* EPI THRESHOLDING *******
THRESHOLD = 6000

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 3
TRACK = True
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = None # Set as None to view all (slow)


# ******* FEATURE EXTRACTION *******
BATCH_SIZE = 100
PLOT_FEATURES = True
TRACKS_PLOT = True
SHOW_EATING = True

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


