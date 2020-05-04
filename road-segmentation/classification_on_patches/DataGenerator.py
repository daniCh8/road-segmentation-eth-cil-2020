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
        self.treshold = 0.25
        
        counter = int((val_split)*len(training_images))
        for i, t in zip(training_images, training_truths):
            if counter == 0 and val_split != 0.0:
                tot_size = int((self.images[0].shape[0]*self.images[0].shape[1])/256)
                self.validation_images, self.validation_labels = self.generate_one_batch(batch_size= tot_size)
                self.images = []
                self.truths = []
            
            counter -= 1
            self.images.append(cv2.imread(image_path + i))
            self.truths.append(rgb2gray(cv2.imread(groundtruth_path + t)))
        
        albumentator = Albumentator()
        albumented = albumentator.albumentate(self.images, self.truths)
        self.images.extend(albumented[0])
        self.truths.extend(albumented[1])
    
    def generate_random_patch(self, big_patch_size= 80, little_patch_size= 16, pad_size= 32):
        index = np.random.randint(len(self.images))
        image = self.images[index]
        groundtruth = self.truths[index]
        
        starting_point_x = np.random.randint(image.shape[0] - big_patch_size)
        starting_point_y = np.random.randint(image.shape[0] - big_patch_size)

        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))

        image_patch = padded_image[starting_point_x:starting_point_x+big_patch_size, starting_point_y:starting_point_y+big_patch_size]
        groundtruth_patch = groundtruth[starting_point_x:starting_point_x+little_patch_size, starting_point_y:starting_point_y+little_patch_size]
        return image_patch, groundtruth_patch
    
    def labelize(self, mask):
        df = np.mean(mask)
        if df > self.treshold:
            return 1
        else:
            return 0
        
    def generate_one_batch(self, batch_size= 16):
        batch = self.generate_batch(batch_size, training_mode= False)
        ls = []
        for b in batch:
            ls.append(b)
        return ls[0]
    
    def generate_batch(self, batch_size= 16, training_mode= True):
        repeating = True
        
        while repeating == True:
            if not training_mode:
                repeating = False
            batch_images = []
            batch_truths = []
        
            for i in range(batch_size):
                img, trt = self.generate_random_patch()
                batch_images.append(img)
                batch_truths.append(self.labelize(trt))
        
            yield np.array(batch_images), np.array(batch_truths)
    
    def return_validation_set(self):
        return self.validation_images, self.validation_labels