import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from keras.layers import Dropout
from cv import kfold_cv

np.random.seed(10)

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

# epochs to try
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# epochs, train error, test error
epoch_errors = []
for e in epochs:
    epoch_errors.append(kfold_cv(5, e, train_X_normalized, train_Y, 32))

batches = [32, 64, 128, 256]
batch_errors = []
for b in batches:
    batch_errors.append(kfold_cv(5, 40, train_X_normalized, train_Y, b))

for x in epoch_errors:
    print(x)

# Train the final model on the entire training set
final_model = Sequential()
final_model.add(Input(shape=(104,)))
final_model.add(Dense(units=72, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(units=48, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(units=24, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(1, kernel_initializer='normal'))
final_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the final model on the entire training set
final_model.fit(train_X_normalized, train_Y, batch_size=32, epochs=21, verbose=0)

# Make predictions on the test set using the final trained model
test_predictions = final_model.predict(test_X_normalized)
TestingData = pd.DataFrame(data=test_X_normalized, columns=test_X_dropped.columns)
TestingData['Predicted Survival Months'] = test_predictions
print(TestingData.head())
