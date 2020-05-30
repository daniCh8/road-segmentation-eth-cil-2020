import keras
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from DataGenerator import DataGenerator
from TestData import TestData
from Labelizer import Labelizer
from visualization import display_predictions
from utils import preprocess_test_images, merge_predictions, submission_outputs
from nets.ures_xception import create_model as ures_xception_net
from nets.u_xception import create_model as u_xception_net
from nets.uspp_xception import create_model as uspp_xception_net
from nets.deepresunet import create_model as deepures_net
from nets.ures_resnet50v2 import create_model as ures_resnet50v2_net
from nets.u_resnet50v2 import create_model as u_resnet50v2_net
from nets.uspp_resnet50v2 import create_model as uspp_resnet50v2_net
from nets.dunet import create_model as du_net

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def create_callbacks(loss='bce', checkpoint_path='../checkpoints/model_checkpoint.h5', with_val=False):
    if with_val:
        monitor_prefix = 'val_'
    else:
        monitor_prefix = ''
    if loss == 'bce':
        to_monitor = monitor_prefix+'acc'
    else:
        to_monitor = monitor_prefix+'dice_coef'
    reduce_lr = ReduceLROnPlateau(monitor=to_monitor, patience=3, mode='max', factor=0.5)
    early_stop = EarlyStopping(monitor=to_monitor, mode='max', patience=10)
    #checkpoint = ModelCheckpoint(checkpoint_path, monitor=to_monitor, mode='max', save_best_only=True) #checkpoints usually slow down the process

    return [reduce_lr, early_stop]#, checkpoint]

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)

    return iou

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

class NNet:
    def __init__(self, val_split=.0, model_to_load='None', net_type='u_xception', load_weights='None'):
        assert net_type in ['u_xception', 'ures_xception', 'uspp_xception', 'deepuresnet', 'u_resnet50v2',  'ures_resnet50v2', 'uspp_resnet50v2', 'dunet'], "net_type must be one of ['u_xception', 'ures_xception', 'uspp_xception', 'deepuresnet', 'u_resnet50v2',  'ures_resnet50v2', 'uspp_resnet50v2', 'dunet']"
        self.net_type = net_type
        if model_to_load == 'None':
            if net_type == 'u_xception':
                self.model = u_xception_net()
            elif net_type == 'ures_xception':
                self.model = ures_xception_net()
            elif net_type == 'uspp_xception':
                self.model = uspp_xception_net()
            elif net_type == 'deepuresnet':
                self.model = deepures_net()
            elif net_type == 'u_resnet50v2':
                self.model = u_resnet50v2_net()
            elif net_type == 'ures_resnet50v2':
                self.model = ures_resnet50v2_net()
            elif net_type == 'uspp_resnet50v2':
                self.model = uspp_resnet50v2_net()
            elif net_type == 'dunet':
                self.model = du_net()
                
            print('created model: {}'.format(self.model.name))
            
            if load_weights != 'None':
                weights = np.load(load_weights, allow_pickle=True)
                self.model.set_weights(weights)
                print('loaded weights: {}'.format(load_weights))
        else:
            self.model = load_model(model_to_load, custom_objects= {'soft_dice_loss':soft_dice_loss, 'dice_coef':dice_coef, 'iou_coef':iou_coef})
            print('loaded model: {}'.format(model_to_load))
            print('model name: {}'.format(self.model.name))
            
        self.data = None
        self.valid_set = None
        self.val_split = val_split
        self.load_data(val_split=val_split)
        
        self.test_data_gen = TestData()
        self.test_images = self.test_data_gen.get_test_data()
        self.preprocessed_test_images = preprocess_test_images(self.test_images)/255
        self.test_images_predictions = None
    
    def load_data(self, val_split=.0):
        self.val_split = val_split
        self.data = DataGenerator(val_split=val_split)
        if self.val_split != .0:
            self.valid_set = self.data.return_validation_set()
        else:
            self.valid_set = next(self.data.generator(len(self.data.images)))
        
    def train(self, loss='bce', epochs=100, l_rate=.0001, batch_size=8, train_on='competition_data'):
        assert loss in ['bce', 'dice'], "loss must be one of ['bce', 'dice']"
        assert train_on in ['competition_data', 'google_data'], "train_on must be one of ['competition_data', 'google_data']"
        optimizer = keras.optimizers.adam(l_rate)
        metrics = ['acc', iou_coef, dice_coef]
        if loss == 'bce':
            self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        else:
            self.model.compile(optimizer=optimizer, loss=soft_dice_loss, metrics=metrics)
        
        if train_on == 'competition_data':
            steps = len(self.data.images) // batch_size
        
            if self.val_split != .0:
                self.model.fit_generator(generator=self.data.generator(batch_size), validation_data=self.valid_set, epochs=epochs, steps_per_epoch=steps, callbacks=create_callbacks(loss, with_val=True))
            else:
                self.model.fit_generator(generator=self.data.generator(batch_size), epochs=epochs, steps_per_epoch=steps, callbacks=create_callbacks(loss))
                
        else:
            steps = len(self.data.additional_images) // batch_size
            #val_data = (np.array(self.data.images)/255, np.round(np.expand_dims(np.array(self.data.truths), -1)/255))
            self.model.fit_generator(generator=self.data.additional_generator(batch_size), epochs=epochs, steps_per_epoch=steps, callbacks=create_callbacks(loss))
        
    def check_outputs(self):
        plt.figure(figsize= (15, 15))
        gen = self.data.generator(30)
        batch = next(gen)
    
        display_predictions(batch[0], self.model.predict(batch[0]), batch[1])
    
    def evaluate_model(self):
        labelizer = Labelizer()
        val_predictions = self.model.predict(self.valid_set[0]).reshape(-1, 400, 400,)
        predictions_labs = labelizer.make_submission(val_predictions)[0]
        groundtruths = labelizer.make_submission(self.valid_set[1])[0]
        print(accuracy_score(groundtruths, predictions_labs))
        
    def save_model(self, path=None):
        if path == None:
            path = "model-{}.npy".format(self.net_type)
        weights = self.model.get_weights()
        np.save(path, weights)
    
    def predict_test_data(self):
        predictions = self.model.predict(self.preprocessed_test_images)
        self.test_images_predictions = merge_predictions(predictions.reshape(-1, 400, 400,), mode='max')
        return self.test_images_predictions
    
    def display_test_predictions(self, submission_path, samples_number=5):
        plt.figure(figsize= (15, 15))
        display_predictions(self.test_images, self.test_images_predictions, submission_outputs=submission_outputs(submission_path, self.test_data_gen.numbers), samples=samples_number)
    
    def create_submission_file(self, path='submission.csv', treshold=.25):
        labelizer = Labelizer(treshold)
        submission = labelizer.make_submission(self.test_images_predictions, self.test_data_gen.numbers)
        submission_df = pd.DataFrame({'id': submission[1], 'prediction': submission[0]})
        submission_df.to_csv(path, index=False)
        return submission_df