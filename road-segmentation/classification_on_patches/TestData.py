import cv2
import os
import functools
import operator
import re
import numpy as np

class TestData:
    def __init__(self, image_path= '../data/test_images/'):
        test_names = os.listdir(image_path)

        self.images = []
        self.numbers = []

        for name in test_names:
            self.images.append(cv2.imread(image_path + name))
            self.numbers.append(int(re.sub("[^0-9]", "", name)))
            
    
    def patchize(self, image, number, pad_offset= 80, patch_size= 80, step= 16, pad_size= 32):
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)))

        patches = []
        sub_names = []
        for j in range(0, padded_image.shape[1] - pad_offset + 1, step):
            for i in range(0, padded_image.shape[0] - pad_offset + 1, step):
                patches.append(padded_image[i:i+patch_size, j:j+patch_size])
                sub_names.append("{:03d}_{}_{}".format(number, j, i))

        return patches, sub_names
    
    
    def return_test_set(self):
        test_data = []
        submission_names = []
        for image, number in zip(self.images, self.numbers):
            patches, sub_names = self.patchize(image, number)
            test_data.extend(patches)
            submission_names.extend(sub_names)

        return test_data, submission_names