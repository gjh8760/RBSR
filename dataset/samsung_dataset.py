import os, glob
import numpy as np
from dataset.base_image_dataset import BaseImageDataset
from data.image_loader import opencv_loader


class Samsung(BaseImageDataset):
    def __init__(self, root=None, image_loader=opencv_loader, initialize=True):
        root = '/data/gjh8760/datasets/SamsungMX2025' if root is None else root
        super().__init__('Samsung', root, image_loader)
        
        if initialize:
            self.initialize()
        
    def initialize(self):
        root = self.root
        self.img_pth = root
        self.image_list = self._get_image_list()
    
    def _get_image_list(self):
        image_list = sorted(os.listdir(self.root))
        return image_list
    
    def get_image_info(self, im_id):
        return {}
    
    def _get_image(self, im_id):
        path = os.path.join(self.img_pth, self.image_list[im_id])
        img = self.image_loader(path)
        return img
    
    def get_image(self, im_id, info=None):
        frame = self._get_image(im_id)

        if info is None:
            info = self.get_image_info(im_id)

        return frame, info

        