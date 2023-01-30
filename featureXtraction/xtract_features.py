import cv2

def get_sift_descriptor(gray_img):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray_img, None)
    return descriptors


def get_pca_components(gray_img):
    pass


def get_wavelet_components(gray_img):
    pass

def get_lda_components(gray_img):
    pass


def get_features(list_img_gr, algorithm='sift'):
     
     match algorithm:
        case "sift":
            features = [get_sift_descriptor(img) for img in list_img_gr]
            return features
        case "pca":
            features = [get_pca_components(img) for img in list_img_gr]
            return features
        case "lda":
            features = [get_lda_components(img) for img in list_img_gr]
            return features
        case "wavelet":
            features = [get_wavelet_components(img) for img in list_img_gr]
            return features
        case default:
            features = [get_sift_descriptor(img) for img in list_img_gr]
            return features

