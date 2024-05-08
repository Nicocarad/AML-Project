from torchvision import transforms

# hue_t = transforms.ColorJitter(hue=0.2)
# gs_t = transforms.Grayscale(3)
hflip_t = transforms.RandomHorizontalFlip(p=1)
rp_t = transforms.RandomPerspective(p=1, distortion_scale=0.5)
rot_t = transforms.RandomRotation(degrees=90)

bright_t = transforms.ColorJitter(brightness=[1, 2])
contrast_t = transforms.ColorJitter(contrast=[2, 5])
saturation_t = transforms.ColorJitter(saturation=[1, 3])

aug_colors = transforms.Compose([bright_t, contrast_t, saturation_t])
aug_positions = transforms.Compose([hflip_t, rp_t, rot_t])
