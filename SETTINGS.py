from pathlib import Path

# ******* GENERAL *******
DIRECTORY = Path('03_test')
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}

# ******* EPI THRESHOLDING *******
THRESHOLD = 6000

# ******* TRACKING *******
OVERLAP_THRESHOLD = 0.2
FRAME_MEMORY = 3
TRACK = False
VIEW_TRACKS = True # Save labelled tracked images
NUM_FRAMES_TO_VIEW = 100 # Set as None to view all (slow)

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


