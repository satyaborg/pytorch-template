from torchvision import transforms, utils
import numpy as np
import random
import math
import PIL

def transformations(img_size):
    """Transformations and augmentations"""
    crop_height, crop_width = 10, 10
    return dict(
            train=transforms.Compose([
                transforms.Lambda(lambda x: foo(x)), # normalize to be [0-255]
                transforms.ToPILImage(), # covert to PIL image
                transforms.RandomCrop(size=(crop_height, crop_width), 
                                    pad_if_needed=True, 
                                    ), # randomly crops 5 secs
                transforms.Resize(size=(img_size, img_size), 
                                interpolation=PIL.Image.NEAREST
                                ), # resize image to be [img_size, img_size]
                transforms.ToTensor(), # normalizes values to be [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            valid=transforms.Compose([
                transforms.ToPILImage(), # covert to PIL image
                transforms.CenterCrop(size=(crop_height, crop_width)),
                transforms.Resize(size=(img_size, img_size), 
                                interpolation=PIL.Image.NEAREST
                                ), # resize image to be [img_size, img_size]
                transforms.ToTensor(), # normalizes values to be [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
