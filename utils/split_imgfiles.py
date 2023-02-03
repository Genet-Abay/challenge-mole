import os
import numpy as np
import shutil

def split_images(img_path, op_path):
    #make train test and validation directories in output folder
    os.makedirs(op_path +'/train')
    os.makedirs(op_path +'/validation')
    os.makedirs(op_path +'/test')


    list_images = os.listdir(img_path)
    np.random.shuffle(list_images)
    train_imgs, val_imgs, test_imgs = np.split(np.array(list_images), [int(len(list_images)*0.7), int(len(list_images)*0.85)])


    train_img_filenames = [img_path+'/'+ name for name in train_imgs.tolist()]
    val_img_filenames = [img_path+'/' + name for name in val_imgs.tolist()]
    test_img_filenames = [img_path+'/' + name for name in test_imgs.tolist()]

    print('Total images: ', len(list_images))
    print('Training: ', len(train_img_filenames))
    print('Validation: ', len(val_img_filenames))
    print('Testing: ', len(test_img_filenames))

    # Copy-pasting images
    for name in train_img_filenames:
        shutil.copy(name, op_path + "/train")

    for name in val_img_filenames:
        shutil.copy(name, op_path + "/validation")

    for name in test_img_filenames:
        shutil.copy(name, op_path + "/test")


