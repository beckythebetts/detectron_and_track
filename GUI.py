import h5py
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import torch
import sys

import mask_funcs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gui:
    def __init__(self, hdf5dataset):
        self.get_data(hdf5dataset)
        self.get_merged_images()
        self.get_tracked_images()
        self.current_image_index = 0
        self.images = self.tracked

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Timelapse")
        self.root.geometry('500x600')
        self.root.resizable(width=True, height=True)

        # Create a label to display images
        self.label_img = tk.Label(self.root)
        self.label_img.pack(fill=tk.BOTH, expand=True)

        # Create a slider
        self.slider = tk.Scale(self.root, from_=0, to=len(self.images) - 1, orient="horizontal", command=self.update_image)
        self.slider.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind the resize event
        self.root.bind("<Configure>", self.on_resize)

        # Initialize with the first image
        self.update_image(self.slider.get())

        # Run the GUI
        self.root.mainloop()

    def update_image(self, val):
        self.current_image_index = int(val)
        img = Image.fromarray(self.images[self.current_image_index], 'RGB')
        # Resize the image to fit the window size
        img = self.resize_image_to_fit(img)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=256)

        img_tk = ImageTk.PhotoImage(img)
        self.label_img.config(image=img_tk)
        self.label_img.image = img_tk

    def on_resize(self, event):
        # Update image when window is resized
        self.update_image(self.slider.get())

    def resize_image_to_fit(self, image):
        max_width = max(self.root.winfo_width(), 1)  # Ensure non-zero dimensions
        max_height = max(self.root.winfo_height(), 1)  # Ensure non-zero dimensions
        img_width, img_height = image.size
        ratio = min(max_width / img_width, max_height / img_height)

        # Calculate new dimensions ensuring they're greater than zero
        new_width = max(int(img_width * ratio), 1)
        new_height = max(int(img_height * ratio), 1)

        return image.resize((new_width, new_height))

    def get_data(self, dataset):
        with h5py.File(dataset, 'r') as f:
            self.phase_data = np.array([f['Images']['Phase'][frame][:]
                                   for frame in list(f['Images']['Phase'].keys())], dtype='uint8')
            self.epi_data = np.array([f['Segmentations']['Epi'][frame][:]
                                 for frame in list(f['Segmentations']['Epi'].keys())], dtype='uint8')
            self.segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
                                               for frame in list(f['Segmentations']['Phase'].keys())], dtype='int16')
            self.max_cell_index = 0
            for frame in f['Segmentations']['Phase']:
                frame = np.array(f['Segmentations']['Phase'][frame])
                max = np.max(frame)
                if max > self.max_cell_index:
                    self.max_cell_index = max

    def get_merged_images(self):
        self.epi_data[self.epi_data > 0] = 255
        epi_channel = make_rgb(self.epi_data)
        epi_channel[:, :, :, 1:3] = 0

        self.merged = ((make_rgb(self.phase_data).astype(np.float32) + epi_channel.astype(np.float32)) / (2)).astype(
                np.uint8)

    def get_tracked_images(self):
        colour_dict = {cell_index: torch.tensor(np.random.uniform(0, (2**8)-1, size=3)).to(device) for cell_index in np.arange(1, self.max_cell_index+1)}
        rgb_phase = make_rgb((self.phase_data))
        self.tracked = np.zeros(rgb_phase.shape)
        for i, (phase_image, segmentation) in enumerate(zip(torch.tensor(rgb_phase).to(device), torch.tensor(self.segmentation_data).to(device))):
            sys.stdout.write(
                f'\rReading frame {i + 1}')
            sys.stdout.flush()
            for cell_index in torch.unique(segmentation)[1:]:
                outline = mask_funcs.mask_outline(torch.where(segmentation==cell_index.item(), 1, 0), thickness=3)
                for c in range(3):
                    phase_image[:, :, c] = torch.where(outline, colour_dict[cell_index.item()][c], phase_image[:, :, c])
            self.tracked[i] = phase_image.cpu().numpy()

def make_rgb(greyscale_im):
    return np.stack((greyscale_im, greyscale_im, greyscale_im), axis=-1)

def main():
    my_gui = Gui(str(Path('Datasets') / 'danhighres' / 'dan10.h5'))
    my_gui.create_gui()

if __name__ == '__main__':
    main()




