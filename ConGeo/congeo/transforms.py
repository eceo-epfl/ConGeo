import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import torch
import numpy as np
from torchvision.transforms import Resize


class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")  

class Zoomin(ImageOnlyTransform):
    def __init__(self, img_size):
         super(Zoomin, self).__init__()
         self.img_size = img_size

    def apply(self, x, **params):
        ratio = random.uniform(1.0, 1.5)
        torch_tensor_resize = Resize([int(ratio*self.img_size[0]), int(ratio*self.img_size[1])])
        resized_tensor = torch_tensor_resize(x)
        
        return resized_tensor   

class LimitedFoV(ImageOnlyTransform):
    def __init__(self, fov=360.):
        super(LimitedFoV, self).__init__(fov)
        self.fov = fov

    def apply(self, x, **params):
        #print(x.shape)
        if self.fov > 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            fov_index = int(self.fov / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
            return img_shift[:,:,:fov_index]
        else:
            return x

class LimitedFoVPad(ImageOnlyTransform):
    def __init__(self, fov=360.):
        super(LimitedFoVPad, self).__init__(fov)
        self.fov = fov

    def apply(self, x, **params):
        #print(x.shape) # 3, h, w
        if self.fov == 361.0: 
            angle = random.randint(0, 359)
            rand_fov = random.randint(180, 360)
            rotate_index = int(angle / 360. * x.shape[2])
            fov_index = int(rand_fov/ 360. * x.shape[2])
            angle2 = random.randint(0, 359)
            roll_index = int(angle2 / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
            img_shift = img_shift[:,:,:fov_index]  
            img_pad = torch.zeros([x.shape[0], x.shape[1], x.shape[2]-fov_index])
            pad_img_shift = torch.cat((img_shift, img_pad), dim=2)
            rolled_img_shift = torch.roll(pad_img_shift, shifts=roll_index, dims=2)
            return rolled_img_shift                     
        elif self.fov > 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            fov_index = int(self.fov / 360. * x.shape[2])
            angle2 = random.randint(0, 359)
            roll_index = int(angle2 / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
            img_shift = img_shift[:,:,:fov_index]  
            img_pad = torch.zeros([x.shape[0], x.shape[1], x.shape[2]-fov_index])
            pad_img_shift = torch.cat((img_shift, img_pad), dim=2)
            rolled_img_shift = torch.roll(pad_img_shift, shifts=roll_index, dims=2)
            return rolled_img_shift
        else:
            return x


class ShiftFoV(ImageOnlyTransform):
    def __init__(self, shift=0):
        super(ShiftFoV, self).__init__()
        self.shift = shift

    def apply(self, x, **params):
        #print(x.shape)
        if self.shift == 0:
            angle = random.randint(0, 359)
            rotate_index = int(angle / 360. * x.shape[2])
            if rotate_index > 0:
                img_shift = torch.zeros(x.shape)
                img_shift[:,:,:rotate_index] = x[:,:,-rotate_index:]
                img_shift[:,:,rotate_index:] = x[:,:,:(x.shape[2] - rotate_index)]
            else:
                img_shift = x
        else:
            img_shift = x
            
        return img_shift


def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
                
            
               
    return satellite_transforms, ground_transforms


def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0,
                       fov=0.0,
                       rotate=False,
                       mask_ratio=0.0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    
 

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                  ])
            
               
    return satellite_transforms, ground_transforms

def get_transforms_train_congeo(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0,
                         fov=180):
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])

    satellite_transforms_con = A.Compose([
                                      A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*image_size_sat[0]),
                                                               max_width=int(0.2*image_size_sat[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*image_size_sat[0]),
                                                               min_width=int(0.1*image_size_sat[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])    
    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])

    ground_transforms_con = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*img_size_ground[0]),
                                                            max_width=int(0.2*img_size_ground[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1*img_size_ground[0]),
                                                            min_width=int(0.1*img_size_ground[0]),
                                                            p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   LimitedFoV(fov=fov),
                                   #LimitedFoVPad(fov=fov),
                                   ])
                
    return satellite_transforms, satellite_transforms_con, ground_transforms, ground_transforms_con

