import h5py
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

class Gui:
    def __init__(self, hdf5dataset):
        self.images = self.get_images(hdf5dataset)
        self.current_image_index = 0

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

    def get_images(self, dataset):
        with h5py.File(dataset, 'r') as f:
            phase_data = np.array([f['Images']['Phase'][frame][:]
                                   for frame in list(f['Images']['Phase'].keys())[:20]], dtype='uint8')
            epi_data = np.array([f['Segmentations']['Epi'][frame][:]
                                 for frame in list(f['Segmentations']['Epi'].keys())[:20]], dtype='uint8')
        epi_data[epi_data > 0] = 255
        epi_channel = make_rgb(epi_data)
        epi_channel[:, :, :, 1:3] = 0
        print('EPiData max: ',epi_channel.max())
        print('Phasedata max: ', phase_data.max())

        merged = ((make_rgb(phase_data).astype(np.float32) + epi_channel.astype(np.float32)) / (2)).astype(
                np.uint8)
        print(merged.shape)
        print(merged[0, 0, 0], merged[0, 0, 1])
        return merged

def make_rgb(greyscale_im):
    return np.stack((greyscale_im, greyscale_im, greyscale_im), axis=-1)

def main():
    my_gui = Gui(str(Path('Datasets') / 'danhighres' / 'dan3.h5'))
    my_gui.create_gui()

if __name__ == '__main__':
    main()




