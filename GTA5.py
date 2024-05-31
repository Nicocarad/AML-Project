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
import os
from torchvision import transforms
import numpy as np
import random
from data_augmentation import DataAugmentation
import matplotlib.pyplot as plt


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def process_directory(root, mode, subfolder, file_suffix):
    result = {}
    file_names = []
    path = osp.join(root, mode, subfolder)
    files = glob.glob(path + "/*" + file_suffix)
    for file in files:
        name = osp.splitext(osp.basename(file))[0]
        result[name] = file
        file_names.append(name)
    return result, file_names


to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def convert_labels(lb_map, label):
    label = np.array(label, dtype=int)
    label_id = lb_map[label[:, :, 0], label[:, :, 1], label[:, :, 2]]
    return label_id


class GTA5(Dataset):
    def __init__(self, root, labels_info, mode, apply_transform):
        super(GTA5, self).__init__()

        assert mode in ("train", "val", "test")
        seed =42
        random.seed(seed)
        np.random.seed(seed)
        self.mode = mode
        self.apply_transform = apply_transform
        self.counter = 0

        if self.apply_transform == True:
            print("Data Augmentation activated")
            self.transform = DataAugmentation()

        self.lb_map = np.zeros((256, 256, 256), dtype=np.int64)
        for el in labels_info:
            color = el["color"]
            trainId = el["trainId"]
            self.lb_map[color[0], color[1], color[2]] = trainId

        self.imgs, img_file_names = process_directory(root, mode, "images", ".png")
        self.labels, fine_file_names = process_directory(root, mode, "labels", ".png")

        assert set(img_file_names) == set(fine_file_names)
        self.img_file_names_filtered = img_file_names

    def __getitem__(self, idx):
        img_name = self.img_file_names_filtered[idx]
        img_path = self.imgs[img_name]
        label_path = self.labels[img_name]

        img = pil_loader(img_path)
        label = pil_loader(label_path)

        if self.mode == "train":
            resize_img = transforms.Resize((512, 1024), interpolation=Image.BILINEAR)
            resize_label = transforms.Resize((512, 1024), interpolation=Image.NEAREST)

            img = resize_img(img)
            label = resize_label(label)

        if self.apply_transform == True and random.uniform(0, 1) > 0.5:

            img, label = self.transform.Positionaltransform(img, label)
            img = self.transform.Colortransform(img)
            

        img = to_tensor(img)

        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = np.squeeze(label)

        label = convert_labels(self.lb_map, label)
        label = label[np.newaxis, :]

        return img, label

    def __len__(self):
        return len(self.img_file_names_filtered)


if __name__ == "__main__":
    from tqdm import tqdm

    with open("./GTA5_info.json", "r") as fr:
        labels_info = json.load(fr)
    print("Load dataset")
    ds = GTA5("./GTA5", labels_info, mode="train", apply_transform=True)
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(set(uni))
