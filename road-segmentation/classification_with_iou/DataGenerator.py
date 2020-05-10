import os
import cv2
import numpy as np
from skimage.color import rgb2gray
import random
from Albumentator import Albumentator

class DataGenerator:
    def __init__(self, image_path= '../data/training/images/', groundtruth_path = '../data/training/groundtruth/', val_split= 0.1):
        training_images = os.listdir(image_path)
        training_truths = os.listdir(groundtruth_path)
        
        #shuffling the images so to obtain a random train-test split
        zipped = list(zip(training_images, training_truths))
        random.shuffle(zipped)
        training_images, training_truths = zip(*zipped)
        
        self.images = []
        self.truths = []
        self.validation_images = []
        self.validation_truths = []
        self.treshold = 0.25
        
        counter = int((val_split)*len(training_images))
        for i, t in zip(training_images, training_truths):
            if counter > 0 and val_split != 0.0:
                self.validation_images.append(cv2.imread(image_path + i))
                self.validation_truths.append(rgb2gray(cv2.imread(groundtruth_path + t)))
            
            counter -= 1
            self.images.append(cv2.imread(image_path + i))
            self.truths.append(rgb2gray(cv2.imread(groundtruth_path + t)))
        
        albumentator = Albumentator()
        albumented = albumentator.albumentate(self.images, self.truths)
        self.images.extend(albumented[0])
        self.truths.extend(albumented[1])
    
    def return_data_set(self):
        return self.images, self.truths
    
    def return_validation_set(self):
        return self.validation_images, self.validation_truths