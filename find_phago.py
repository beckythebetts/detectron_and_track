from pathlib import Path
import torch

import SETTINGS
import utils

def main():
    phase = [f for f in (SETTINGS.DIRECTORY / 'tracked' / 'phase').iterdir()]
    epi = [f for f in (SETTINGS.DIRECTORY / 'tracked' / 'epi').iterdir()]

    for phase_frame, epi_frame in zip(phase, epi):
        phase_frame = torch.tensor(utils.read_tiff(phase_frame))
        epi_frame = torch.tensor(utils.read_tiff(epi_frame))
        intersection = torch.logical_and(phase_frame>0, epi_frame>0)
if __name__ == '__main__':
    main()