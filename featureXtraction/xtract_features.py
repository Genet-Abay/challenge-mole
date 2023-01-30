import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



'''Notes
The SIFT family of interest point detectors. Includes things like SURF, FREAK, BRISK, etc. Primarily used for object recognition, tracking and registration.
HoG and it's variants. Used for object detection, handwritten digit recognition.
LBP and variants. Previously used for face recognition and texture recognition applications.
GIST and variants. Introduced for scene recognition and analysis. Similar in spirit to HoG and SIFT.
How could we miss out the big daddy of features these days? Features learned on CNN for object recognition and detection.
'''

def get_sift_descriptor(gray_img):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray_img, None)
    return descriptors


def get_surf_features(gray_img):
    pca = PCA(n_components=30)
    img_comp = pca.fit_transform(gray_img)
    return img_comp


def get_hog_features(gray_img):
    pass

def get_brisk_features(gray_img):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    x_new = lda.transform(X) 

def get_freak_features(gray_img):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    x_new = lda.transform(X) 


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

