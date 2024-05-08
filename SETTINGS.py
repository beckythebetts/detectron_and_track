from pathlib import Path

DIRECTORY = Path('03')
CLASSES = {'phase': 'Amoeba', 'epi': 'Yeast'}

# EPI THRESHOLDING
THRESHOLD = 6000

# TRACKING
OVERLAP_THRESHOLD = 0.5
FRAME_MEMORY = 3
TRACK = True
VIEW_TRACKS = True # Save labelled tracked images

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


