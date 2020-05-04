from TestData import TestData

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Sequential
import keras

import numpy as np
import pandas as pd

data = TestData()
test_images, submission_names = data.return_test_set()

model = tf.keras.models.load_model('model.h5')
predictions = model.predict(np.array(test_images))
rounded_preds = np.round(predictions).reshape(-1).astype(int)

submission_df = pd.DataFrame({'id': submission_names, 'prediction': rounded_preds})
submission_df.to_csv('submission.csv', index= False)