import numpy as np
from albumentations import VerticalFlip, HorizontalFlip, RandomRotate90, ShiftScaleRotate

class Albumentator:
    def __init__(self, p= .8):
        self.proba = .8
        
        self.h_flipper  = HorizontalFlip(p = p)
        self.v_flipper  = VerticalFlip(p= p)
        self.r_rotater  = RandomRotate90(p= p)
        self.s_rotater  = ShiftScaleRotate(p= p)
        
    def albumentate(self, images, truths):
        i_albumentated = []
        t_albumentated = []
        
        for i, t in zip(images, truths):
            h = self.h_flipper(image= i, mask= t)
            v = self.v_flipper(image= i, mask= t)
            r = self.r_rotater(image= i, mask= t)
            s = self.s_rotater(image= i, mask= t)

            if not np.allclose(h['image'], i):
                i_albumentated.append(h['image'])
                t_albumentated.append(h['mask'])

            if not np.allclose(v['image'], i):
                i_albumentated.append(v['image'])
                t_albumentated.append(v['mask'])

            if not np.allclose(r['image'], i):
                i_albumentated.append(r['image'])
                t_albumentated.append(r['mask'])

            if not np.allclose(s['image'], i):
                i_albumentated.append(s['image'])
                t_albumentated.append(s['mask'])
        
        return i_albumentated, t_albumentated