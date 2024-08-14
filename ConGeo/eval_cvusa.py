import os
import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader
from congeo.dataset.cvusa import CVUSADatasetEval
from congeo.transforms import get_transforms_val
from congeo.evaluate.cvusa_and_cvact import evaluate
from congeo.model import TimmModel,TimmModel_ConGeo

@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    
    # Dataset
    data_folder = "./data/CVUSA/CVPR_subset"     
    
    checkpoint_start = ''
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    fov: float = 0.0 # 0.0 for north-aligned, others for limited FoV or random shiftting
    rotate: bool = False
    mask_ratio: float = 0.0
    random_fov = False

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel_ConGeo(config.model,
                      pretrained=True,
                      img_size=config.img_size,
                      random_fov=config.random_fov)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    fov = config.fov
    rotate = config.rotate
    mask_ratio = config.mask_ratio

    image_size_sat = (img_size, img_size)
    
    new_width = config.img_size * 2    
    new_hight = round((224 / 1232) * new_width)
    #image_size_sat = (new_hight, new_width)
    img_size_ground = (new_hight, new_width)
     
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
        
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               fov=fov,
                                                               rotate=rotate,
                                                               mask_ratio=mask_ratio,
                                                               )

    # Reference Satellite Images
    # for polar
    '''
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="polar_reference",
                                              transforms=sat_transforms_val,
                                              )
    '''
    reference_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )


    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    # Query Ground Images Test
    query_dataset_test = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",    #change here to evaluate the performance under different noises: gaussian_query, motion_query
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "CVUSA", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)