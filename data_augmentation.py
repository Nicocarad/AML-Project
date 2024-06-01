from torchvision import transforms
import random


class DataAugmentation:
    def __init__(self):
        self.hflip = transforms.functional.hflip
        self.color_jitter = transforms.ColorJitter(
            brightness=[2, 2], contrast=[2, 2], saturation=[2, 2]
        )

    def Positionaltransform(self, img, label):
        img, label = self.hflip(img), self.hflip(label)
        return img, label

    def Colortransform(self, image):

        return self.color_jitter(image)
