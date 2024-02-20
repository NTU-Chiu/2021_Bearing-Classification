import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm

# Create a directory
def create_directory(path):
  if not os.path.exists(path):
    os.makedirs(path)

def load_data(path):
  train_x = sorted(glob(os.path.join(path,'train','images','*.tif'))) # training photo
  train_y = sorted(glob(os.path.join(path,'train','mask','*.gif'))) # training mask

  test_x = sorted(glob(os.path.join(path,'test','images','*.tif'))) # training photo
  test_y = sorted(glob(os.path.join(path,'test','mask','*.gif'))) # training mask

  return ((train_x, train_y), (test_x,test_y))

def augment_data(images, masks, save_path, augment=True):
  size = (512, 512)
  
  for idx, (x,y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
    # 抽出必要的字詞
    name = x.split('/')[-1].split('.')[0] # ['', 'content', 'dataset', 'train', 'images', '21_training.tif'] 負一 只取最後一項
    
    # read image and mask
    x = cv2.imread(x,cv2.IMREAD_COLOR) # (584, 565, 3)
    y = imageio.mimread(y)[0] # (584, 565)

    # 處理augmentation
    if augment == True:
      aug = HorizontalFlip(p=1.0)
      augmented = aug(image=x, mask=y)
      x1 = augmented['image']
      y1 = augmented['mask']

      aug = VerticalFlip(p=1.0)
      augmented = aug(image=x, mask=y)
      x2 = augmented['image']
      y2 = augmented['mask']

      aug = Rotate(limit=45,p=1.0) # degree for rotation !!!
      augmented = aug(image=x, mask=y)
      x3 = augmented['image']
      y3 = augmented['mask']

      X = [x,x1,x2,x3]
      Y = [y,y1,y2,y3]

      pass

    else:
      X = [x]  ### ????
      Y = [y]

    # 重新命名放入new_data資料夾
    index = 0
    for i,j in zip(X,Y):
      i = cv2.resize(i,size) # 改變大小
      j = cv2.resize(j,size)

      tmp_img = f'{name}_{index}.png' # 21_training_0.png
      tmp_mask = f'{name}_{index}.png'

      image_path = os.path.join(save_path,'images',tmp_img) 
      mask_path = os.path.join(save_path,'mask',tmp_mask)

      cv2.imwrite(image_path, i) # 寫入資料夾
      cv2.imwrite(mask_path, j)

      index +=1
