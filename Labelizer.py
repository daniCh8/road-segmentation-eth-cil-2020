import numpy as np

class Labelizer:
    def __init__(self, treshold=.25):
        self.treshold = treshold
        
    def patchize(self, pred, number, patch_size= 16, step= 16):
        labels  = []
        numbers = []
        for j in range(0, pred.shape[1], step):
            for i in range(0, pred.shape[0], step):
                labels.append(self.labelize(pred[i:i+patch_size, j:j+patch_size]))
                numbers.append("{:03d}_{}_{}".format(number, j, i))

        return labels, numbers
    
    def make_submission(self, predictions, numbers= None):
        sub_numbers = []
        sub_labels  = []
        
        if(numbers == None):
            numbers = range(len(predictions))
        
        for pred, number in zip(predictions, numbers):
            labels, names = self.patchize(pred, number)
            sub_labels.extend(labels)
            sub_numbers.extend(names)
        
        return sub_labels, sub_numbers
    
    def labelize(self, mask):
        df = np.mean(mask)
        if df > self.treshold:
            return 1
        else:
            return 0