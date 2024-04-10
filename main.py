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
layers = [1, 2, 3]
layers_errors = []
errors = []

for num in layers:
    layers_errors.append(kfold_cv(5, 10, train_X_normalized, train_Y, 32, num))

smallest_test_error_index = 0
for i in range(len(layers_errors)):
    if layers_errors[i][2] < layers_errors[smallest_test_error_index][2]:
        smallest_test_error_index = i

best_num_layers = layers_errors[smallest_test_error_index][4]
for b in batches:
    for e in epochs:
        errors.append(kfold_cv(5, e, train_X_normalized, train_Y, b, best_num_layers))

smallest_test_error_index = 0
for i in range(len(errors)):
    if errors[i][2] < errors[smallest_test_error_index][2]:
        smallest_test_error_index = i

best_num_epochs = errors[smallest_test_error_index][0]
best_num_batches = errors[smallest_test_error_index][3]

# Train the final model on the entire training set
final_model = Sequential()
final_model.add(Input(shape=(104,)))
final_model.add(Dense(units=72, activation='relu'))
final_model.add(Dropout(0.2))
if best_num_layers >= 2:
    final_model.add(Dense(units=48, activation='relu'))
    final_model.add(Dropout(0.2))
    if best_num_layers == 3:
        final_model.add(Dense(units=24, activation='relu'))
        final_model.add(Dropout(0.2))
final_model.add(Dense(1, kernel_initializer='normal'))
final_model.compile(loss='mean_squared_error', optimizer='adam')

# Train the final model on the entire training set
final_model.fit(train_X_normalized, train_Y, batch_size=best_num_batches, epochs=best_num_epochs, verbose=0)

# Make predictions on the test set using the final trained model
test_predictions = final_model.predict(test_X_normalized)
TestingData = pd.DataFrame(data=test_X_normalized, columns=test_X_dropped.columns)
TestingData['Predicted Survival Months'] = test_predictions
tests_last_col = TestingData['Predicted Survival Months']
tests_last_col.to_csv('TestingData.csv', index=False)
print("Epochs Used: ", best_num_epochs)
print("Batch Size Used: ", best_num_batches)
print("Number of Hidden Layers: ", best_num_layers)
print(TestingData.head())

# Constructing the DataFrame
columns = ['Epochs', 'Batch Size', 'Training Error', 'Testing Error']
data = []

for error in errors:
    epochs, train_error, test_error, batch_size, _ = error
    data.append([epochs, batch_size, train_error, test_error])

df = pd.DataFrame(data, columns=columns)

# Print or manipulate the DataFrame as needed
print(df)

# Constructing the DataFrame for layers_errors
layer_columns = ['Number of Layers', 'Training Error', 'Testing Error']
layer_data = []

for error in layers_errors:
    epochs, train_error, test_error, batch_size, num_layers = error
    layer_data.append([num_layers, train_error, test_error])

layer_df = pd.DataFrame(layer_data, columns=layer_columns)

# Print or manipulate the DataFrame as needed
print(layer_df)

# Saving the DataFrame to CSV files
layer_df.to_csv('layers_errors.csv', index=False)
df.to_csv('epochs_errors.csv', index=False)
