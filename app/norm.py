import cv2
import numpy as np
from dataload import load_tracks
from constants import *

def compute_mean_std(image_filenames):
    # Initialize sums and squares of sums for mean and std computation
    n_pixels = 0
    sum_channels = np.zeros(3)
    sum_squares_channels = np.zeros(3)

    for filename in image_filenames:
        img = cv2.imread(filename)  # By default, images are read in BGR format
        img = img / 255.0  # Normalize to [0,1]

        sum_channels += np.sum(img, axis=(0,1))
        sum_squares_channels += np.sum(np.square(img), axis=(0,1))
        
        # Increment the total pixel count
        n_pixels += img.shape[0] * img.shape[1]

    # Compute mean and standard deviation
    mean = sum_channels / n_pixels
    std = np.sqrt((sum_squares_channels / n_pixels) - np.square(mean))

    # Return as RGB instead of BGR
    return mean[::-1], std[::-1]

data = load_tracks()

image_filenames = [track[ALBUM_IMG_KEY] for track in data]

mean, std = compute_mean_std(image_filenames)
print("Mean:", mean)
print("StdDev:", std)
