import os
import time
from glob import glob
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model_Unet import build_unet
from model_VGG import VGG16
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time

def train(model, loader, optimizer, loss_fn, device):
  epoch_loss = 0.0

  model.train() # not test, not evaluate
  for x, y in loader:
    x = x.to(device, dtype = torch.float32)
    y = y.to(device, dtype = torch.float32)

    # Sets the gradients of all optimized torch.Tensor s to zero.
    # 每次更新weights,bias後歸零
    optimizer.zero_grad() 
    y_prediction = model(x)
    loss = loss_fn(y_prediction, y)
    loss.backward()
    optimizer.step()
    epoch_loss +=loss.item()
    # print(y.shape) # 2,1,512,512
    # print(y_prediction.shape) # 2,1,512,512

  epoch_loss = epoch_loss/len(loader)

  return epoch_loss

def evaluate(model, loader, loss_fn, device):
  epoch_loss = 0.0

  model.eval()
  with torch.no_grad():
    for x, y in loader:
     x = x.to(device, dtype = torch.float32)
     y = y.to(device, dtype = torch.float32)
     
     y_prediction = model(x)
     loss = loss_fn(y_prediction, y)
     epoch_loss +=loss.item()
    
    epoch_loss = epoch_loss/len(loader)

  return epoch_loss

if __name__=="__main__":
    # fixed
    seeding(42)

    # file
    create_dir('C:/Users/User/Desktop/UNet/files')


    # create_dir('/content/files')

    # Load data
    train_x = sorted(glob("C://Users//User//Desktop//UNet//new_data//train//images//*"))
    train_y = sorted(glob("C://Users//User//Desktop//UNet//new_data//train//mask//*"))
    valid_x = sorted(glob("C://Users//User//Desktop//UNet//new_data//test//images//*"))
    valid_y = sorted(glob("C://Users//User//Desktop//UNet//new_data//test//mask//*"))

    # Check data size
    data_str = f"Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}"
    print(data_str)

    # Hyperparameters
    H,W = 512,512
    size = (H,W)
    batch_size = 2
    num_epoch = 50
    lr =  1e-4 # UNet 1e-4 # VGG 1e-3
    checkpoint_path = "C:/Users/User/Desktop/UNet/files/checkpoint.pth"

    # Dataset and Loader
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
      dataset = train_dataset,
      batch_size = batch_size,
      shuffle = True,
      num_workers = 0
    )


    valid_loader = DataLoader(
      dataset = valid_dataset,
      batch_size = batch_size,
      shuffle = False,
      num_workers = 0
    )

    # device = torch.device('cpu') # 'cuda'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # loss_fn = BCELoss()
    loss_fn = DiceLoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = DiceBCELoss()


    loss_list_train =[]
    loss_list_val =[]


    # Strat Training
    best_valid_loss = float('inf')
    for epoch in range(num_epoch):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        loss_list_train.append(train_loss)
        loss_list_val.append(valid_loss)


        # Save better model
        if valid_loss < best_valid_loss:
            data_str = f'Valid Loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}'
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {mins}m {secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.5f}'
        data_str += f'\tVal Loss: {valid_loss:.5f}\n'
        print(data_str)
      
