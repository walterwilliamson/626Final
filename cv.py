from sklearn.model_selection import KFold
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
import tensorflow as tf



# Perform k-fold cross-validation
def kfold_cv(num_folds, epochs, training_data, outcome_data, batch, layers):
    np.random.seed(10)
    tf.random.set_seed(10)
    # Initialize KFold object
    kf = KFold(n_splits=num_folds)
    # Lists to store MSE for each fold
    train_errors = []
    test_errors = []

    for train_index, val_index in kf.split(training_data):
        # Split data into training and validation sets for this fold
        X_train_fold, X_val_fold = training_data.iloc[train_index], training_data.iloc[val_index]
        y_train_fold, y_val_fold = outcome_data.iloc[train_index], outcome_data.iloc[val_index]

        # Create a new model for each fold
        model_cv = Sequential()
        model_cv.add(Input(shape=(104,)))
        model_cv.add(Dense(units=72, activation='relu'))
        model_cv.add(Dropout(0.2))
        if layers >= 2:
            model_cv.add(Dense(units=48, activation='relu'))
            model_cv.add(Dropout(0.2))
            if layers == 3:
                model_cv.add(Dense(units=24, activation='relu'))
                model_cv.add(Dropout(0.2))
        model_cv.add(Dense(1, kernel_initializer='normal'))
        model_cv.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model on this fold
        model_cv.fit(X_train_fold, y_train_fold, batch_size=batch, epochs=epochs, verbose=0)

        train_predictions = model_cv.predict(X_train_fold)
        test_predictions = model_cv.predict(X_val_fold)

        train_errors.append(mean_squared_error(y_train_fold, train_predictions))
        test_errors.append(mean_squared_error(y_val_fold, test_predictions))

    # Calculate average MSE across all folds
    avg_test_error = np.mean(test_errors)
    avg_train_error = np.mean(train_errors)

    return [epochs, avg_train_error, avg_test_error, batch, layers]
