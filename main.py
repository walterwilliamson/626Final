import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input


# Load data from files
avg_intensity = pd.read_csv("avg_intensity.csv")
train_data = pd.read_csv("train_data.csv")

# Combine on id column
combined_df = pd.merge(avg_intensity, train_data, on='id', how='inner')

# Check to make sure it worked
print(combined_df)

# Create outcome vector
Y = combined_df['OSmonth']
# Create input vector
X = combined_df.drop(['id', 'OSmonth'], axis=1)

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
model.fit(X, Y, batch_size=5, epochs=300)

# Generating Predictions on testing data
Predictions = model.predict(X)

# Get test data

# For Wednesday

# TestingData = pd.DataFrame(data=test_data, columns=Predictors)
# TestingData['Price'] = y_test_orig
# TestingData['PredictedPrice'] = Predictions
# TestingData.head()
