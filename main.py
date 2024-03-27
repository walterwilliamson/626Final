import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from keras.layers import Dropout
from sklearn.model_selection import KFold


# use function from data_preprocessing.py
data = preprocess_data()
train = data[0]
test = data[1]

# Create outcome vector
train_Y = train['OSmonth']
# Create input vectors
train_X = train.drop(['id', 'OSmonth'], axis=1)
test_X = test

# Normalize input data
scaler = StandardScaler()
# Drop 'id' column from both training and testing data
test_X_dropped = test_X.drop('id', axis=1)
train_X_normalized = scaler.fit_transform(train_X)
test_X_normalized = scaler.transform(test_X_dropped)

# Convert normalized arrays back to DataFrame
train_X_normalized = pd.DataFrame(train_X_normalized, columns=train_X.columns)
test_X_normalized = pd.DataFrame(test_X_normalized, columns=test_X_dropped.columns)

# Define the number of folds for cross-validation
num_folds = 5

# Initialize KFold object
kf = KFold(n_splits=num_folds)

# Lists to store MSE for each fold
mse_scores = []

# Perform k-fold cross-validation
for train_index, val_index in kf.split(train_X_normalized):
    # Split data into training and validation sets for this fold
    X_train_fold, X_val_fold = train_X_normalized.iloc[train_index], train_X_normalized.iloc[val_index]
    y_train_fold, y_val_fold = train_Y.iloc[train_index], train_Y.iloc[val_index]

    # Create a new model for each fold
    model_cv = Sequential()
    model_cv.add(Input(shape=(104,)))
    model_cv.add(Dense(units=72, activation='relu'))
    model_cv.add(Dropout(0.01))
    model_cv.add(Dense(units=48, activation='relu'))
    model_cv.add(Dropout(0.01))
    model_cv.add(Dense(units=24, activation='relu'))
    model_cv.add(Dropout(0.01))
    model_cv.add(Dense(1, kernel_initializer='normal'))
    model_cv.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model on this fold
    model_cv.fit(X_train_fold, y_train_fold, batch_size=10, epochs=50, verbose=0)

    # Evaluate the model on the validation set for this fold
    y_pred_fold = model_cv.predict(X_val_fold)
    mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
    mse_scores.append(mse_fold)

# Calculate average MSE across all folds
avg_mse = np.mean(mse_scores)
print('Average MSE across all folds:', avg_mse)

# Train the final model on the entire training set
final_model = Sequential()
final_model.add(Input(shape=(104,)))
final_model.add(Dense(units=72, activation='relu'))
final_model.add(Dropout(0.01))
final_model.add(Dense(units=48, activation='relu'))
final_model.add(Dropout(0.01))
final_model.add(Dense(units=24, activation='relu'))
final_model.add(Dropout(0.01))
final_model.add(Dense(1, kernel_initializer='normal'))
final_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the final model on the entire training set
final_model.fit(train_X_normalized, train_Y, batch_size=10, epochs=50)

# Make predictions on the test set using the final trained model
test_predictions = final_model.predict(test_X_normalized)
TestingData = pd.DataFrame(data=test_X_normalized, columns=test_X_dropped.columns)
TestingData['Predicted Survival Months'] = test_predictions
print(TestingData.head())

"""
#################################################################
# Tests we can run for the training data model (the one we use to test stuff)
# Predict survival months for the training data
train_predictions = model.predict(train_X_normalized).flatten()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(train_Y, train_predictions)
print('Mean Squared Error (MSE): %.2f' % mse)

# chatgpt custom accuracy check

# Define a threshold for acceptable error
threshold = 1  # for example, ±1 month

# Calculate the absolute difference between predicted and actual survival months
absolute_diff = abs(train_predictions - train_Y)

# Count the number of predictions within the threshold
within_threshold = (absolute_diff <= threshold).sum()

# Calculate accuracy as the percentage of predictions within the threshold
accuracy = (within_threshold / len(train_Y)) * 100
print('Accuracy within ±{} month(s): {:.2f}%'.format(threshold, accuracy))

# This prints loss not accuracy because this is a regression model rather than a classification model
# (according to chatgpt)
print(model.evaluate(train_X_normalized, train_Y))
"""
