#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image 
import os.path as osp
import os
from PIL import Image
import numpy as np
import json



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')




class CityScapes(Dataset):
    def __init__(self, root, mode = "train" ):
        super(CityScapes, self).__init__()
        
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        print('self.mode', self.mode)
        
        
        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        
        
        ## parse img directory
        self.imgs = {}
        img_file_names = []
        img_path = osp.join(root, 'images', mode) # cityscapes/images/train
        cities_img_folders = os.listdir(img_path)
        for city_img_folder  in cities_img_folders:
            city_path = osp.join(img_path, city_img_folder) # cityscapes/images/train/city
            img_names = os.listdir(city_path)
            filtered_names = [el.replace('_leftImg8bit.png', '') for el in img_names]
            img_path_list = [osp.join(city_path, el) for el in img_names] # list of path like this: cityscapes/images/train/city/hanover_000000_000019
            img_file_names.extend(filtered_names) # list of names like this: hanover_000000_000019
            self.imgs.update(dict(zip(filtered_names, img_path_list))) # dictionary of names and paths

        ## parse gt directory
        self.labels = {}
        fine_file_names = []
        fine_path = osp.join(root, 'gtFine', mode) # cityscapes/gtFine/train or cityscapes/gtFine/val
        cities_fine_folders = os.listdir(fine_path)
        for city_fine_folder in cities_fine_folders:
            city_path = osp.join(fine_path, city_fine_folder) # cityscapes/gtFine/train/[city]
            img_names = os.listdir(city_path)
            img_names = [el for el in img_names if 'labelTrainIds' in el]
            filtered_names = [el.replace('_gtFine_labelTrainIds.png', '') for el in img_names]
            img_path_list = [osp.join(city_path, el) for el in img_names]
            fine_file_names.extend(filtered_names)
            self.labels.update(dict(zip(filtered_names, img_path_list)))

    def __getitem__(self, idx):
        # TODO
        

    def __len__(self):
        # TODO
