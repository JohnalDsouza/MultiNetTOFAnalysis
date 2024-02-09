
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, MeanSquaredLogarithmicError
from Simple_CNN.src.data import *



# Load Train, Valid, Test Data

X_train, X_val, y_train, y_val = train_val_data(train_path = "dataset/Foamboard/complex_data_train.xlsx",
                                                 label_path = "dataset/Foamboard/raml_train_label.xlsx")



X_train.shape,X_val.shape




# Define the input
inputs = Input(shape=(17, 1))

# First Convolutional Layer
x = Conv1D(filters=96, kernel_size=3, strides=1, activation='relu')(inputs)

# Max Pooling
x = MaxPooling1D(pool_size=2, strides=2)(x)

# Second Convolutional Layer
x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)

# Third Convolutional Layer
x = Conv1D(filters=384, kernel_size=2, activation='relu')(x)

# Fourth Convolutional Layer
x = Conv1D(filters=384, kernel_size=2, activation='relu')(x)

# Fifth Convolutional Layer
x = Conv1D(filters=256, kernel_size=2, activation='relu')(x)

# Max Pooling
x = MaxPooling1D(pool_size=2, strides=2)(x)

# Flatten the output of the convolutional layers
x = Flatten()(x)


x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(4096, activation='relu')(x)

x = Dropout(0.5)(x)

# Second Fully Connected Layer
x = Dense(4096, activation='relu')(x)

x = Dropout(0.5)(x)

# Third Fully Connected Layer
x = Dense(1000, activation='relu')(x)

x = Dropout(0.5)(x)


x = Dense(128, activation='relu')(x) 
 
x = Dense(64, activation='relu')(x)

x = Dense(32, activation='relu')(x)

x = Dense(16, activation='relu')(x)


outputs = Dense(1, activation='linear')(x) 


# Create the model
model = Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=0.0001) 
model.compile(loss=MeanSquaredError(), optimizer= optimizer, metrics=['mae','mse'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)


# Display the model's architecture
model.summary()



print("TRAINING STARTED")

history = model.fit(X_train, y_train, epochs=100, batch_size=64,
                    validation_data=(X_val, y_val), 
                    callbacks=[checkpoint, reduce_lr, early_stopping])



# Plot training & validation loss values
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('Simple_CNN/CNN_plot/loss_plot.png')
plt.show()


