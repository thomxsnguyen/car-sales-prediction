import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Loading the Dataset using pandas '

car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding="ISO-8859-1")

# Seperating features (input) and labels (outputs) 

X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
y = car_df['Car Purchase Amount']

# Scaling the Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = np.array(y).reshape(-1, 1)  
y_scaled = scaler.fit_transform(y)

# Splits the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25)

# Building the Neural Network model 

model = Sequential()
model.add(Dense(5, input_dim=5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Algorithim that adjusts the weights and biases of the neural netowrk

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model    x-train: input, y-train: output, epochs: number of iterations, batch_size: number of samples per gradient update, verbose: controls level of detail printed
# validation_split: fraction of the training data to be used as validation data, basically a reward system forn the model

epochs_hist = model.fit(X_train, y_train, epochs=10, batch_size=50, verbose=1, validation_split=0.2)

# Plotting the loss function

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

X_test_sample = np.array([[1, 50, 50000, 10000, 600000]])
X_test_sample_scaled = scaler.transform(X_test_sample)
prediction = model.predict(X_test_sample_scaled)

prediction_original_scale = scaler.inverse_transform(prediction)
print(f"Predicted Car Purchase Amount: {prediction_original_scale}")
