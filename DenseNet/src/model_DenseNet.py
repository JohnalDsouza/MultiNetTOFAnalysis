
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Concatenate, Add, Activation, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, MeanSquaredLogarithmicError
from InceptionNet.src.data import *



# Load Train, Valid, Test Data

X_train, X_val, y_train, y_val = train_val_data(train_path = "dataset/Foamboard/complex_data_train.xlsx",
                                                 label_path = "dataset/Foamboard/raml_train_label.xlsx")


def dense_block(x, num_conv, growth_rate):
    for _ in range(num_conv):
        # Directly use Conv1D followed by Activation without BatchNormalization
        x1 = Conv1D(filters=growth_rate, kernel_size=3, padding='same')(x)
        x1 = Activation('relu')(x1)
        
        # Concatenate input and output
        x = Concatenate()([x, x1])
    return x

# Define Transition Layer without BatchNormalization
def transition_layer(x, reduction_rate):
    # Directly reduce feature maps without BatchNormalization
    reduced_filters = int(tf.keras.backend.int_shape(x)[-1] * reduction_rate)
    x = Conv1D(filters=reduced_filters, kernel_size=1, padding='same')(x)
    
    # Perform average pooling
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    return x

inputs = Input(shape=(17, 1))

# Initial convolution
x = Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)

# Add dense blocks and transition layers
x = dense_block(x, num_conv=4, growth_rate=12)
x = transition_layer(x, reduction_rate=0.5)

x = dense_block(x, num_conv=4, growth_rate=12)
x = transition_layer(x, reduction_rate=0.5)

# Final part of the model without BatchNormalization
x = Activation('relu')(x)

# Flatten and define the rest of the model
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(8, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)



# Create the model
model = Model(inputs=inputs, outputs=outputs)

optimizer = Adam(learning_rate=0.001) 
model.compile(loss=MeanAbsoluteError(), optimizer= optimizer, metrics=['mae'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)


# Display the model's architecture
plot_model(model, to_file='model_summary.png', show_shapes=True)



print("TRAINING STARTED")

history = model.fit(X_train, y_train, epochs=150, batch_size=128,
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
plt.savefig('loss_plot.png')
plt.show()


