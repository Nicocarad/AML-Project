from torchvision import transforms
import random
class DataAugmentation:
    def __init__(self):
        self.hflip = transforms.functional.hflip
        self.color_jitter = transforms.ColorJitter(brightness=[1, 2], contrast=[2, 3], saturation=[1, 3])

    def Positionaltransform(self, img, label):
        random_value = random.uniform(-90, 90)
        img, label = img.rotate(random_value), label.rotate(random_value)
        img, label = self.hflip(img), self.hflip(label)
        return img, label

    def Colortransform(self, image):
        return self.color_jitter(image)
