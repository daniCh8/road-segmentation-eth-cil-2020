from DataGenerator import DataGenerator

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Sequential
import keras

data = DataGenerator()

inputs = Input((80, 80, 3))
s = Lambda(lambda x: x / 255) (inputs)

def convLevel(filters, kernel_size, dropout, input_vec):
  conv = Conv2D(filters, kernel_size, activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (input_vec)
  conv = BatchNormalization() (conv)
  conv = Dropout(dropout) (conv)
  conv = Conv2D(filters, kernel_size, activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (conv)
  conv = BatchNormalization() (conv)
  return conv

conv1 = convLevel(16, (3, 3), 0.1, inputs)
layer_1 = MaxPooling2D((2, 2)) (conv1)

conv2 = convLevel(32, (3, 3), 0.1, layer_1)
layer_2 = MaxPooling2D((2, 2)) (conv2)

conv3 = convLevel(64, (3, 3), 0.2, layer_2)
layer_3 = MaxPooling2D((2, 2)) (conv3)

conv4 = convLevel(128, (3, 3), 0.2, layer_3)
layer_4 = MaxPooling2D((2, 2)) (conv4)

conv5 = convLevel(256, (3, 3), 0.3, layer_4)

def convLevelTranspose(filters, dropout, input_vec, to_concatenate, axis= -1):
  upsample = Conv2DTranspose(filters, (2, 2), strides= (2, 2), padding= 'same') (input_vec)
  upsample = concatenate([upsample, to_concatenate], axis= axis)

  conv = Conv2D(filters, (3, 3), activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (upsample)
  conv = BatchNormalization() (conv)
  conv = Dropout(dropout) (conv)
  conv = Conv2D(filters, (3, 3), activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (conv)
  conv = BatchNormalization() (conv)
  return conv

conv6 = convLevelTranspose(128, 0.2, conv5, conv4)
conv7 = convLevelTranspose(64, 0.2, conv6, conv3)
conv8 = convLevelTranspose(32, 0.2, conv7, conv2)
conv9 = convLevelTranspose(16, 0.2, conv8, conv1, 3)

unet = Conv2D(1, (1, 1), activation= 'elu') (conv9)
flat = Flatten() (unet)

dense1 = Dense(128) (flat)
lrelu1 = LeakyReLU(alpha= 0.1) (dense1)
dropout1 = Dropout(0.25) (lrelu1)

outputs = Dense(1, activation= "sigmoid") (dropout1)

model = Model(inputs= [inputs], outputs= [outputs])
model.compile(Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit_generator(data.generate_batch(512), validation_data= data.return_validation_set(), steps_per_epoch= 128, epochs= 12)
model.save("model.h5", overwrite=True, include_optimizer=False)