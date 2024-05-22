from torchvision import transforms
import random


class DataAugmentation:
    def __init__(self):
        self.hflip = transforms.functional.hflip
        self.color_jitter = transforms.ColorJitter(
            brightness=[2, 3], contrast=[1, 3], saturation=[2, 3]
        )
        # self.hue_t = transforms.ColorJitter(hue=0.2)
        # self.gs_t = transforms.Grayscale(3)

    def Positionaltransform(self, img, label):
        random_value = random.uniform(-90, 90)
        img, label = img.rotate(random_value), label.rotate(random_value)
        img, label = self.hflip(img), self.hflip(label)
        return img, label

    def Colortransform(self, image):

        # image = self.hue_t(image)
        # image = self.gs_t(image)
        return self.color_jitter(image)
