#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image 
import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import glob
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def process_directory(root, mode, subfolder, file_suffix):
    result = {}
    file_names = []
    path = osp.join(root, subfolder, mode)
    city_folders = os.listdir(path)
    for city_folder in city_folders:
        city_path = osp.join(path, city_folder)
        img_names = os.listdir(city_path)
        if subfolder == "gtFine":
            img_names = [el for el in img_names if 'labelTrainIds' in el]
        filtered_names = [el.replace(file_suffix, '') for el in img_names]
        img_path_list = [osp.join(city_path, el) for el in img_names]
        file_names.extend(filtered_names)
        result.update(dict(zip(filtered_names, img_path_list)))
    return result, file_names




def process_directory_optimized(root, mode, subfolder, file_suffix):
    result = {}
    file_names = []
    path = osp.join(root, subfolder, mode, '**', '*' + file_suffix)
    files = glob.glob(path, recursive=True)
    for file in files:
        name = osp.basename(file).replace(file_suffix, '')
        result[name] = file
        file_names.append(name)
    return result, file_names


def convert_labels(lb_map, label):
        for k, v in lb_map.items():
            label[label == k] = v
        return label

# # Define transforms for training phase
# train_transform = transforms.Compose([transforms.Resize(512,1024),      # Resizes short size of the PIL image to 512x1024                                                             
                                      
# ])


to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

class CityScapes(Dataset):
    def __init__(self, root, mode="train", transform=None):
        super(CityScapes, self).__init__(transform=transform)
        
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        print('self.mode', self.mode)
        
        
        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        
        # define two dictionaries that link file names to their full addresses
        
        self.imgs, img_file_names = process_directory(root, mode, 'images', '_leftImg8bit.png')
        self.labels, fine_file_names = process_directory(root, mode, 'gtFine', '_gtFine_labelTrainIds.png')
        
        
        ## parse img directory
        
        # self.imgs = {}
        # self.img_file_names = []
        # img_path = osp.join(root, 'images', mode) # cityscapes/images/train
        # cities_img_folders = os.listdir(img_path)
        # for city_img_folder  in cities_img_folders:
        #     city_path = osp.join(img_path, city_img_folder) # cityscapes/images/train/city
        #     img_names = os.listdir(city_path)
        #     filtered_names = [el.replace('_leftImg8bit.png', '') for el in img_names]
        #     img_path_list = [osp.join(city_path, el) for el in img_names] # list of path like this: cityscapes/images/train/city/hanover_000000_000019
        #     img_file_names.extend(filtered_names) # list of names like this: hanover_000000_000019
        #     self.imgs.update(dict(zip(filtered_names, img_path_list))) # dictionary of names and paths

        # ## parse gtFine directory
        # self.labels = {}
        # fine_file_names = []
        # fine_path = osp.join(root, 'gtFine', mode) # cityscapes/gtFine/train or cityscapes/gtFine/val
        # cities_fine_folders = os.listdir(fine_path)
        # for city_fine_folder in cities_fine_folders:
        #     city_path = osp.join(fine_path, city_fine_folder) # cityscapes/gtFine/train/[city]
        #     img_names = os.listdir(city_path)
        #     img_names = [el for el in img_names if 'labelTrainIds' in el]
        #     filtered_names = [el.replace('_gtFine_labelTrainIds.png', '') for el in img_names]
        #     img_path_list = [osp.join(city_path, el) for el in img_names]
        #     fine_file_names.extend(filtered_names)
        #     self.labels.update(dict(zip(filtered_names, img_path_list))) # dictionary of names and paths

        self.img_file_names_filtered = img_file_names
        
        # Integrity Check
        assert set(img_file_names) == set(fine_file_names)
        assert set(self.img_file_names_filtered) == set(self.imgs.keys())
        assert set(self.img_file_names_filtered) == set(self.labels.keys())

    def __getitem__(self, idx):
        
        img_name  = self.img_file_names_filtered[idx] # get the name of the image file
        img_path = self.imgs[img_name] # get the path of the image file from the dictionary key: img_name value: path of the image
        label_path = self.labels[img_name] # get the path of the label file from the dictionary key: img_name value: path of the label
        
        img = pil_loader(img_path)
        label = Image.open(label_path)
        
        # Applies preprocessing when accessing the image
        if self.transform is not None:
           
            img_lb_dict = dict(img = img, label = label)
            img_lb_dict = self.transform(img_lb_dict)
            img, label = img_lb_dict['img'], img_lb_dict['label']
            
            
        img = to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = convert_labels(self.lb_map,label)
        
        return img, label

    def __len__(self):
        return len(self.img_file_names_filtered)
        


if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('/content/Cityspaces/', mode='train')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
# Da capire:
# 1. Applicare la normalizzazione o no?
# 2. Dobbiamo usare le immagini in bianco e nero o quelle a colori
