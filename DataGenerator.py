import os
from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
import random
from albumentations import VerticalFlip, HorizontalFlip, RandomRotate90, ShiftScaleRotate
from tqdm import tqdm

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
        print('Reading images...')
        for i, t in tqdm(list(zip(training_images, training_truths))):
            if counter > 0 and val_split != 0.0:
                self.validation_images.append(imread(image_path + i))
                self.validation_truths.append(rgb2gray(imread(groundtruth_path + t)))
            
            counter -= 1
            self.images.append(imread(image_path + i))
            self.truths.append(rgb2gray(imread(groundtruth_path + t)))
        print('Done!')
            
        self.albument_p = .5
        self.albumenters = [VerticalFlip(p=self.albument_p),
                            HorizontalFlip(p=self.albument_p),
                            RandomRotate90(p=self.albument_p),
                            ShiftScaleRotate(p=self.albument_p)]
    
    def generator(self, batch_size= 16):
        while True:
            indices = np.random.choice(len(self.images), batch_size, replace= False)
            batch_x = []
            batch_y = []
            for i in indices:
                albumented = self.albumenters[np.random.randint(len(self.albumenters))] (image=self.images[i], mask=self.truths[i])
                batch_x.append(albumented['image'])
                batch_y.append(albumented['mask'])
            yield np.array(batch_x)/255, np.round(np.expand_dims(np.array(batch_y), -1)/255)
            
    def return_validation_set(self):
        return np.array(self.validation_images)/255, np.round(np.expand_dims(np.array(self.validation_truths), -1)/255)