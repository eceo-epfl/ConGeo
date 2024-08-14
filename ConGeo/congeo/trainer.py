import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list



def train_contrast_congeo(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference1, reference2, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference1 = reference1.to(train_config.device)
                reference2 = reference2.to(train_config.device)
            
                # Forward pass
                features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp())

                    
                else:
                    loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp())
                    loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                    loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                    loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())

                loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4 
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference1 = reference1.to(train_config.device)
            reference2 = reference2.to(train_config.device)
            
            # Forward pass
            features_q1, features_q2, features_r1, features_r2 = model(query1, query2, reference1, reference2)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss1 = loss_function(features_q1, features_r1, model.module.logit_scale.exp())
                loss2 = loss_function(features_q1, features_q2, model.module.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.module.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.module.logit_scale.exp())

            else:
                loss1 = loss_function(features_q1, features_r1, model.logit_scale.exp()) 
                loss2 = loss_function(features_q1, features_q2, model.logit_scale.exp())
                loss3 = loss_function(features_r1, features_r2, model.logit_scale.exp())
                loss4 = loss_function(features_r1, features_q2, model.logit_scale.exp())

                
            loss = loss1+0.5*loss2+0.5*loss3+0.25*loss4

            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg




    # set model train mode
    model.train()
    '''
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        print(k)
    '''
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query1, query2, reference, ids in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query1 = query1.to(train_config.device)
                query2 = query2.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                query1, query2, reference1 = model(query1, query2, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss1 = loss_function(query1, reference1, model.module.logit_scale.exp())
                    loss2 = loss_function(query2, reference1, model.module.logit_scale.exp())
                else:
                    loss1 = loss_function(query1, reference1, model.logit_scale.exp()) 
                    loss2 = loss_function(query2, reference1, model.logit_scale.exp()) 
                loss = loss1+loss2
                losses.update(loss.item())
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            # data (batches) to device   
            query1 = query1.to(train_config.device)
            query2 = query2.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass

            query1, query2, reference1 = model(query1, query2, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss1 = loss_function(query1, reference1, model.module.logit_scale.exp())
                loss2 = loss_function(query2, reference1, model.module.logit_scale.exp())
            else:
                loss1 = loss_function(query1, reference1, model.logit_scale.exp()) 
                loss2 = loss_function(query2, reference1, model.logit_scale.exp()) 
            loss = loss1+loss2

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg