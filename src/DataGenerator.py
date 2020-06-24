from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
from albumentations import VerticalFlip, HorizontalFlip, RandomRotate90, ElasticTransform, RandomContrast, HueSaturationValue, RandomBrightness
from tqdm import tqdm
import os
import random 


class DataGenerator:
    def __init__(self, image_path='../../data/training/images/', groundtruth_path='../../data/training/groundtruth/', val_split=0.1, additional_images_path='../../additional_data_generation/additional_data/images/', additional_masks_path='../../additional_data_generation/additional_data/masks/'):
        training_images = os.listdir(image_path)
        training_truths = os.listdir(groundtruth_path)
        
        # shuffling the images so to obtain a random train-test split
        zipped = list(zip(training_images, training_truths))
        random.shuffle(zipped)
        training_images, training_truths = zip(*zipped)
        
        self.images = []
        self.truths = []
        self.validation_images = []
        self.validation_truths = []
        self.additional_images = []
        self.additional_masks = []
        self.treshold = 0.25
        
        counter = int((val_split)*len(training_images))
        print('Reading images...', flush=True)
        for i, t in tqdm(list(zip(training_images, training_truths))):
            if counter > 0 and val_split != 0.0:
                self.validation_images.append(imread(image_path + i))
                self.validation_truths.append(rgb2gray(imread(groundtruth_path + t)))
            
            counter -= 1
            self.images.append(imread(image_path + i))
            self.truths.append(rgb2gray(imread(groundtruth_path + t)))
        print('Done!')
        
        additional_paths = [p for p in list(os.listdir(additional_images_path)) if p.endswith('.png')]
        random.shuffle(additional_paths)
        print('Reading additional data...', flush=True)
        for p in tqdm(additional_paths):
            self.additional_images.append(imread(additional_images_path + p))
            self.additional_masks.append(rgb2gray(imread(additional_masks_path + p)))
        print('Done!')
            
        self.albument_p = .5
        self.albumenters_1 = [VerticalFlip(p=self.albument_p),
                              HorizontalFlip(p=self.albument_p),
                              RandomRotate90(p=self.albument_p),
                              ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=self.albument_p)]
        
        self.albumenters_2 = [RandomContrast(limit=.6, p=self.albument_p),
                              HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=self.albument_p),
                              RandomBrightness(limit=0.2, p=self.albument_p)]
    
    def albument_pair(self, img, msk):
        albumented_1 = self.albumenters_1[np.random.randint(len(self.albumenters_1))] (image=img, mask=msk)
        albumented_2 = self.albumenters_2[np.random.randint(len(self.albumenters_2))] (image=albumented_1['image'], mask=albumented_1['mask'])
        return albumented_2
    
    def generator(self, batch_size=16):
        while True:
            indices = np.random.choice(len(self.images), batch_size, replace=False)
            batch_x = []
            batch_y = []
            for i in indices:
                albumented = self.albument_pair(img=self.images[i], msk=self.truths[i])
                batch_x.append(albumented['image'])
                batch_y.append(albumented['mask'])
            yield np.array(batch_x)/255, np.round(np.expand_dims(np.array(batch_y), -1)/255)
            
    def additional_generator(self, batch_size= 16):
        while True:
            indices = np.random.choice(len(self.additional_images), batch_size, replace=False)
            batch_x = []
            batch_y = []
            for i in indices:
                albumented = self.albument_pair(img=self.additional_images[i], msk=self.additional_masks[i])
                batch_x.append(albumented['image'])
                batch_y.append(albumented['mask'])
            yield np.array(batch_x)/255, np.round(np.expand_dims(np.array(batch_y), -1)/255)
            
    def return_validation_set(self):
        return np.array(self.validation_images)/255, np.round(np.expand_dims(np.array(self.validation_truths), -1)/255)