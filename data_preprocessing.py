import pandas as pd


def preprocess_data():
    # importing the training data for merging
    intensity = pd.read_csv("avg_intensity.csv")
    blob_intensity = pd.read_csv("blob_intensity.csv")

    train_intensity = intensity[0:224]
    test = intensity[225:281]

    train_blob_intensity = blob_intensity[0:224]
    test_blob = blob_intensity[225:281]

    train_outcome = pd.read_csv("train_data.csv")

    # combining training and testing data
    train = pd.merge(train_intensity, train_blob_intensity, on='id', how='inner')
    train = pd.merge(train, train_outcome, on='id', how='inner')
    test = pd.merge(test, test_blob, on='id', how='inner')

    return train, test
