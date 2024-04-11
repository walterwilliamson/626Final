import numpy as np
import pandas as pd
import imageio
import os
from tqdm import tqdm

# Specify these args
img_dir = 'images'  # Directory that saves your images
n_train = 281  # Number of images
n_protein = 52  # Number of proteins

# Initialize the list to store average intensity of top tiles for each protein
avg_list = []

# Process each image
for i in tqdm(range(1, n_train + 1), total=n_train, desc="Processing"):
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
    top_ten_tiles = tiles[:10]

    # Calculate the average intensity for each protein using the ten most intense tiles
    protein_intensities = {"id" : i}
    for j in range(1, n_protein + 1):
        protein_tile_intensities = [tile[:, :, j-1] for tile in top_ten_tiles]
        protein_intensities['protein' + str(j)] = np.mean(protein_tile_intensities)

    # Append to avg_list
    avg_list.append(protein_intensities)

# Convert the list to Data frame
avg_df = pd.DataFrame(avg_list)

# Save to csv
avg_df.to_csv('blob_intensity.csv', index=False)

print(avg_df.head())
