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


# Test the function with image directory
# This can be deleted later or commented out or something it just makes sure the right images go in
# each category
image_dir = 'images'
n_train = 225  # Number of training images
test_train_data, test_test_data = preprocess_data(image_dir, n_train)

print("Training Data:")
print(test_train_data)
print("\nTesting Data:")
print(test_test_data)
