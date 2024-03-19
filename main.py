import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from data_preprocessing import preprocess_data


#use function from data_preprocessing.py
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
model.add(Dense(units=36, activation='tanh'))

# Output layer
model.add(Dense(1, kernel_initializer='normal'))

# Compile
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to training set
model.fit(train_X, train_Y, batch_size=5, epochs=300)

# Generating Predictions on testing data
Predictions = model.predict(X)

# Get test data

# For Wednesday

# TestingData = pd.DataFrame(data=test_data, columns=Predictors)
# TestingData['Price'] = y_test_orig
# TestingData['PredictedPrice'] = Predictions
# TestingData.head()
