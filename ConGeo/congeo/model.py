import torch
import timm
import numpy as np
import torch.nn as nn
import random
from torchvision.transforms import Resize

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
             
            return image_features

class TimmModel_aug(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel_aug, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
            imgq1 = img1
            start = int(imgq1.size(-1)*70/360)
            stop = imgq1.size(-1)
            fov = random.randint(start, stop)
            imgq2 = imgq1[...,:fov]
            image_features1 = self.model(imgq2)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2              
        else:
            image_features = self.model(img1)
             
            return image_features

class TimmModel_ConGeo(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 random_fov=False):
                 
        super(TimmModel_ConGeo, self).__init__()
        
        self.img_size = img_size
        self.random_fov = random_fov
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, imgq1, imgq2=None, imgr1=None, imgr2=None):
        
        if imgq2 is not None:
            if self.random_fov == False:
                image_featuresq1 = self.model(imgq1)     
                image_featuresq2 = self.model(imgq2)
                image_featuresr1 = self.model(imgr1)
                image_featuresr2 = self.model(imgr2)
                
                return image_featuresq1, image_featuresq2, image_featuresr1, image_featuresr2           
            
            else:
                # random fov between 70-360
                random_fov = random.randint(int(imgq2.size(-1)*7/36), imgq2.size(-1))
                imgq2 = imgq2[...,:random_fov]
                image_featuresq1 = self.model(imgq1)     
                image_featuresq2 = self.model(imgq2)
                image_featuresr1 = self.model(imgr1)
                image_featuresr2 = self.model(imgr2)
                
                return image_featuresq1, image_featuresq2, image_featuresr1,  image_featuresr2         
              
        else:
            
            image_features = self.model(imgq1)
             
            return image_features
