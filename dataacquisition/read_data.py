import cv2
import os
import pandas as pd
import os

def get_imgs(path_to_images):
    images = []
    for filename in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images,filename))
        if img is not None:
            images.append(img)
    return images


def get_metadata_dict(metadata_file):
    meta_data = pd.read_csv(metadata_file)
    metadata_dict = dict(zip(meta_data['image_id'], meta_data['dx']))
    return metadata_dict


def get_pixel_data(filename):
    pixel_data = pd.read_csv(filename)
    return pixel_data


# op_path = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images_part1N2_SplitedTT"






