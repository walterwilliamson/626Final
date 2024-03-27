# break into grids and average intense blobs
import numpy as np
import pandas as pd
import imageio
from IPython.display import display, Image
import time
from tqdm import tqdm
import os

# ---------- specify these args ----------
img_dir = 'images'  # dir that saves your images
n_train = 281  # number of images
n_protein = 52  # number of proteins
# ----------------------------------------

avg_list = []
for i in tqdm(np.arange(1, n_train + 1), total=n_train, desc="Processing"):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)

    M = img.shape[0] // 10
    N = img.shape[1] // 10
    tiles = [img[x:x + M, y:y + N] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)]

    tile_averages = []
    for tile in tiles:
        avg_intensity = np.mean(tile)
        tile_averages.append(avg_intensity)

    # Sort based on average intensities of individual tiles
    tile_averages.sort(reverse=True)

    # Take the top ten tiles
    top_ten = tile_averages[:10]

    # Calculate the mean intensity of the top ten tiles
    top_ten_mean = sum(top_ten) / 10

    # Append to avg_list
    avg_list.append([i] + [top_ten_mean] * n_protein)

avg_df = pd.DataFrame(avg_list, columns=['id'] + ['protein' + str(i) for i in range(1, 52 + 1)])
avg_df.to_csv('blob_intensity.csv', index=False)

print(avg_df.head())

# did this work
# did this work part 2
# walter was here
