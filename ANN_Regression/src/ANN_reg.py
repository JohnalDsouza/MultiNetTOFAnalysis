
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, MeanSquaredLogarithmicError
from data import *

# Load Train, Valid, Test Data

X_train, X_val, y_train, y_val = train_val_data(train_path = "Foamboard/complex_data_train.xlsx",
                                                 label_path = "Foamboard/raml_train_label.xlsx")


# Define the model

inputs = Input(shape=(17,1))
x = BatchNormalization()(inputs) 

x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x) 

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)


x = Dense(128, activation='relu')(x) 
x = BatchNormalization()(x) 
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)

outputs = Dense(1, activation='linear')(x) 

# Create the model
model = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(learning_rate=0.0001)

 
model.compile(loss=MeanSquaredError(), optimizer= optimizer, metrics=['mae','mse'])

 
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

print("Training Started")
# Fit the model and save the history
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
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
plt.savefig('ANN_plot/loss_plot.png')
plt.show()