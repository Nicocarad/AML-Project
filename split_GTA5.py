import os
import random
import shutil
from tqdm import tqdm




def get_image_names(directory):
    return [f for f in os.listdir(directory) if f.endswith((".jpg", ".png"))]


def shuffle_vector(vector):
    random.shuffle(vector)
    return vector


def split_vector(vector, size):
    return vector[:size], vector[size:]


def save_images_to_directory(images, source_directory, target_directory):
    os.makedirs(target_directory, exist_ok=True)
    for image in tqdm(
        images,
        desc="Copying images from {} to {}".format(source_directory, target_directory),
        unit="image",
    ):
        shutil.copy(os.path.join(source_directory, image), target_directory)

def main(root="GTA5/GTA5"):
    
    random.seed(42)
    directory = "GTA5/GTA5/images"
    image_names = get_image_names(directory)
    shuffled_vector = shuffle_vector(image_names)
    vector1, vector2 = split_vector(shuffled_vector, 500)


    save_images_to_directory(vector1, "GTA5/GTA5/images", "GTA5/GTA5/val/images")
    save_images_to_directory(vector2, "GTA5/GTA5/images", "GTA5/GTA5/train/images")

    save_images_to_directory(vector1, "GTA5/GTA5/labels", "GTA5/GTA5/val/labels")
    save_images_to_directory(vector2, "GTA5/GTA5/labels", "GTA5/GTA5/train/labels")


if __name__ == "__main__":
    main()