import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from keras.layers import Dropout
from cv import kfold_cv
import keras

keras.utils.set_random_seed(812)

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

# epochs, train error, test error, batch size
batches = [32, 64, 128, 256]
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
errors = []

for b in batches:
    for e in epochs:
        errors.append(kfold_cv(5, e, train_X_normalized, train_Y, b))

smallest_test_error_index = 0
for i in range(len(errors)):
    if errors[i][2] < errors[smallest_test_error_index][2]:
        smallest_test_error_index = i

# Train the final model on the entire training set
final_model = Sequential()
final_model.add(Input(shape=(104,)))
final_model.add(Dense(units=70, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(units=35, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(1, kernel_initializer='normal'))
final_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the final model on the entire training set
final_model.fit(train_X_normalized, train_Y, batch_size=64, epochs=40, verbose=0)

# Make predictions on the test set using the final trained model
test_predictions = final_model.predict(test_X_normalized)
TestingData = pd.DataFrame(data=test_X_normalized, columns=test_X_dropped.columns)
TestingData['Predicted Survival Months'] = test_predictions
tests_last_col = TestingData['Predicted Survival Months']
tests_last_col.to_csv('TestingData.csv', index=False)
print(TestingData.head())

# Constructing the DataFrame
columns = ['Epochs', 'Batch Size', 'Training Error', 'Testing Error']
data = []

for error in errors:
    epochs, train_error, test_error, batch_size = error
    data.append([epochs, batch_size, train_error, test_error])

df = pd.DataFrame(data, columns=columns)

# Saving the DataFrame to CSV files
df.to_csv('epochs_errors.csv', index=False)
