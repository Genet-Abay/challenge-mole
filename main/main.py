
from dataacquisition import read_data, split_imgfiles
from preprocessing import process
from featureXtraction import xtract_features
from model import *

def prepare_data(root_path= r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images_part1N2_SplitedTT"):   
    train_imgs = root_path + "/train"
    val_imgs = root_path + "/validation"
    test_imgs = root_path + "/test"
    
    list_train_imgs = read_data.get_imgs(train_imgs)
    list_train_imgs = process(list_train_imgs)
    features_train_imgs = xtract_features.get_features(list_train_imgs, 'sift')

    list_val_imgs = read_data.get_imgs(val_imgs)
    list_val_imgs = process(list_val_imgs)
    features_val_imgs = xtract_features.get_features(list_val_imgs, 'sift')

    list_test_imgs = read_data.get_imgs(test_imgs)
    list_train_imgs = process(list_test_imgs)
    features_test_imgs = xtract_features.get_features(list_test_imgs, 'sift')

    return features_train_imgs, features_val_imgs, features_test_imgs


def get_trained_model(model_name, train_data, validation_data, test_data):
    match model_name:
        case "keras":
            model = 
            return "keras"
        case "pytorch":
            return "pytorch"
        case "ml_sklrn":
            return "sklrn"
        case default:
            return "keras"
        
    

def main():
    # spliting images in a given img_path folder and put the slplited images in the fiven op_path folder if needed
    # img_path = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images_part1N2"
    # op_path = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images_part1N2_SplitedTT"
    # split_imgfiles.split_images(img_path, op_path)

    train_data, validation_data, test_data = prepare_data()
    model = get_trained_model('keras', train_data, validation_data)
    
    # save model
    print(model)
    return None


if __name__ == '__main__':
    main()