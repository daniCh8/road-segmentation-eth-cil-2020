import cv2
import os
import functools
import operator
import re
import numpy as np

class TestData:
    def __init__(self, image_path= '../../data/test_images/'):
        test_names = os.listdir(image_path)

        self.images = []
        self.numbers = []
        
        self.treshold = .25

        for name in test_names:
            self.images.append(cv2.imread(image_path + name))
            self.numbers.append(int(re.sub("[^0-9]", "", name)))
            
    def get_test_data(self):
        return self.images