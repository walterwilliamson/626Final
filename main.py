import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from data_preprocessing import preprocess_data

# use function from data_preprocessing.py
data = preprocess_data()
train = data[0]
test = data[1]

# Create outcome vector
train_Y = train['OSmonth']
# Create input vectors
train_X = train.drop(['id', 'OSmonth'], axis=1)
test_X = test

# Create empty model
model = Sequential()

# Defining the Input layer
model.add(Input(shape=(52,)))

# model.add(Dense(units=36, input_dim=52, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=36, activation='relu'))
# model.add(Dense(units=36, activation='tanh'))

# Output layer
model.add(Dense(1, kernel_initializer='normal'))

# Compile
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to training set
model.fit(train_X, train_Y, batch_size=5, epochs=300)

# Generating Predictions on testing data - drop id column when passing test_X
# Also not 100% sure this is what we were supposed to past just adding a reminder to
# ask tomorrow haha
Predictions = model.predict(test_X.drop(test_X.columns[0], axis=1))

# For testing
# Predictions = model.predict(train_X)
# TestingData = pd.DataFrame(data=train_X, columns=train_X.columns[1:])

# Get test data

# For Wednesday - We don't know the actual outcomes do we? Just realized that we might
# not need that column. No idea why it's the same number for all of the predictions
TestingData = pd.DataFrame(data=test_X.drop(test_X.columns[0], axis=1), columns=test_X.columns[1:])
TestingData['Predicted Survival Months'] = Predictions
print(TestingData.head())
