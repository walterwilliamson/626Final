import pandas as pd


def preprocess_data():
    # importing the training data for merging
    intensity = pd.read_csv("avg_intensity.csv")

    train_intensity = intensity[0:224]
    test = intensity[225:280]

    train_outcome = pd.read_csv("train_data.csv")

    # combining training data
    train = pd.merge(train_intensity, train_outcome, on='id', how='inner')

    return train, test
