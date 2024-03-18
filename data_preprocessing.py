import os
import pandas as pd


def preprocess_data(image_dir, n_train):
    # List all files in the image directory
    image_files = os.listdir(image_dir)

    # Sort the image files numerically
    image_files.sort(key=lambda x: int(x.split('.')[0]))

    # Extract file names for training and testing
    train_files = image_files[:n_train]
    test_files = image_files[n_train:]

    # Create DataFrame to store file names and labels
    train_data = pd.DataFrame({'image_file': train_files})
    test_data = pd.DataFrame({'image_file': test_files})

    return train_data, test_data


