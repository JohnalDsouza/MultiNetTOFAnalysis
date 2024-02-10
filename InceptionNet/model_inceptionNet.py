
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, MeanSquaredLogarithmicError
from InceptionNet.data import *



# Load Train, Valid, Test Data

X_train, X_val, y_train, y_val = train_val_data(train_path = "dataset/Foamboard/complex_data_train.xlsx",
                                                 label_path = "dataset/Foamboard/raml_train_label.xlsx")



def inception_module(input_layer, filter_operation):
    """
    Creates an Inception module for 1D data.
    
    Parameters:
    - input_layer: Input layer to the module
    - filter_operation: Dictionary containing the number of filters for each operation
    
    Returns:
    - A tensor output from the Inception module
    """
    # Convolution with kernel size 1
    conv1 = Conv1D(filters=filter_operation['conv1'], kernel_size=1, padding='same', activation='relu')(input_layer)
    
    # Convolution with kernel size 1 followed by convolution with kernel size 3
    conv3 = Conv1D(filters=filter_operation['conv3_reduce'], kernel_size=1, padding='same', activation='relu')(input_layer)
    conv3 = Conv1D(filters=filter_operation['conv3'], kernel_size=3, padding='same', activation='relu')(conv3)
    
    # Convolution with kernel size 1 followed by convolution with kernel size 5
    conv5 = Conv1D(filters=filter_operation['conv5_reduce'], kernel_size=1, padding='same', activation='relu')(input_layer)
    conv5 = Conv1D(filters=filter_operation['conv5'], kernel_size=5, padding='same', activation='relu')(conv5)
    
    # MaxPooling followed by Convolution with kernel size 1
    pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_layer)
    pool = Conv1D(filters=filter_operation['pool_proj'], kernel_size=1, padding='same', activation='relu')(pool)
    
    # Concatenate all the operations
    output_layer = concatenate([conv1, conv3, conv5, pool], axis=-1)
    # Match the input dimension to the output using 1x1 convolution if necessary
    input_shape = input_layer.get_shape().as_list()[-1]
    output_shape = output_layer.get_shape().as_list()[-1]
    if input_shape != output_shape:
        input_layer = Conv1D(filters=output_shape, kernel_size=1, padding='same', activation=None)(input_layer)
    
    # Add the input_layer to the output_layer (now that their dimensions match)
    output_layer = Add()([input_layer, output_layer])
    
    return output_layer
    

# Define the input shape
inputs = Input(shape=(17, 1))

# x = Conv1D(32, 3, strides=1, activation='relu', padding='same')(inputs)
x = Conv1D(64, 3, strides=1, activation='relu', padding='same')(inputs)
x = MaxPooling1D(3, strides=2, padding='same')(x)

# Example Inception module application
x = inception_module(x, filter_operation={'conv1': 64, 'conv3_reduce': 96, 'conv3': 128,
                                                'conv5_reduce': 16, 'conv5': 32, 'pool_proj': 32})
x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
x = inception_module(x, filter_operation={'conv1': 64, 'conv3_reduce': 96, 'conv3': 128,
                                                'conv5_reduce': 16, 'conv5': 32, 'pool_proj': 32})
x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)



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
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)


# Display the model's architecture
plot_model(model, to_file='model_summary.png', show_shapes=True)



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
plt.savefig('loss_plot.png')
plt.show()


