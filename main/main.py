from dataacquisition import read_data, split_imgfiles
from preprocessing import process
from featureXtraction import xtract_features
from matplotlib import pyplot as plt
from model import dl_keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import cv2
import os


def prepare_data_splitedfolder(root_path= r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images_part1N2_SplitedTT"):   
    train_imgs = root_path + "/train"
    val_imgs = root_path + "/validation"
    test_imgs = root_path + "/test"
    
    list_train_imgs = read_data.get_imgs(train_imgs)
    list_train_imgs = process(list_train_imgs)
    
    list_val_imgs = read_data.get_imgs(val_imgs)
    list_val_imgs = process(list_val_imgs)
    
    list_test_imgs = read_data.get_imgs(test_imgs)
    list_train_imgs = process(list_test_imgs)    

    return list_train_imgs, list_val_imgs, list_test_imgs

def get_data():
    # Load the image data and metadata
    images_dir = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images"
    df_original = pd.read_csv(r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_metadata.csv")
    df = df_original.sample(n=500)

    # Preprocess the image data
    images = []
    labels = []
    for i in range(len(df)):
        # Load the image
        img_path = os.path.join(images_dir, df.iloc[i]['image_id'] + '.jpg')
        img_path = os.path.join(images_dir, df.iloc[i]['image_id'] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        images.append(img)
        labels.append(df.iloc[i]['dx'])

    images = np.array(images) / 255.0
    labels = np.array(labels)

    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    labels = to_categorical(labels)

    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    return train_images, val_images, train_labels, val_labels


def get_trained_model(model_name, train_img, validation_img, test_label, val_label):
    model = ''
    history = ''
    match model_name:
        case "keras":
            model, history = dl_keras.get_trained_model(train_img, validation_img, test_label, val_label, (100,100), 7, 10)
            return model, history
        case "pytorch":
            model, history = ''
            return model, history
        case "ml_sklrn":
            model, history = ''
            return model, history
        case default:
            model, history = ''
            return model, history
        
    
    
def view_result(history):

     # Plot the training and validation accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(range(1, 11), acc, label='Training Accuracy')
    plt.plot(range(1, 11), val_acc, label='Validation Accuracy')
    plt.legend(loc='best')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(1, 11), loss, label='Training Loss')
    plt.plot(range(1, 11), val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def main():

    epochs = 10
    train_img, validation_img, test_label, val_label = get_data()

    # preprocess with selected process: resize, gray scale, smoothing, sharpening
    # train_data = process.preprocess(train_data, resize = True, sharpen = True)
    # validation_data = process.preprocess(validation_data, resize = True, sharpen = True)
    # test_data = process.preprocess(test_data, resize = True, sharpen = True)
     
    # if featue extraction needed 
    # features_train_imgs = xtract_features.get_features(train_data, 'sift')
    # features_val_imgs = xtract_features.get_features(validation_data, 'sift')
    # features_test_imgs = xtract_features.get_features(test_data, 'sift')
    # model, history = get_trained_model('keras', features_train_imgs, features_val_imgs)

    model, history = get_trained_model('keras',train_img, validation_img, test_label, val_label)
    model.summary()
   
    # save model
    view_result(history, epochs )
  


if __name__ == '__main__':
    main()