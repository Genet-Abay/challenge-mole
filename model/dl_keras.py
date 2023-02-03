import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

def get_data():
    # Load the image data and metadata
    images_dir = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images"
    df_original = pd.read_csv(r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_metadata.csv")
    df = df_original.copy()#.sample(n=5000)
    df['concern'] = df['dx'].apply(lambda x : 0 if ((x == 'nv') | (x == 'bkl') | (x == 'df') | (x == 'vasc')) else 1)

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

    # encoder = LabelEncoder()
    # encoder.fit(labels)
    # labels = encoder.transform(labels)

    # # Convert the labels to categorical data
    # labels = to_categorical(labels)

    # Split the data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    return train_images, val_images, train_labels, val_labels


#augment images
datagen = ImageDataGenerator(
rotation_range=45,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

def create_test_model(train_images, val_images, train_labels, val_labels):
    # Define the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model without augmented data
    history = model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))

    # Train model using the ImageDataGenerator object - augmented images
    # history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=32), steps_per_epoch=len(train_images) / 10, epochs=10, validation_data=(val_images, val_labels))
    
    return model, history

def show_result(history):
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
    train_images, val_images, train_labels, val_labels = get_data()
    model, history = create_test_model(train_images, val_images, train_labels, val_labels)
    show_result(history)

    pred = model.predict(val_images, val_labels)
    report = metrics.classification_report(val_labels, pred)
    print(report)

    # Save the model
    # model.save("../model_keras.h5")

    
    


if __name__ == '__main__':
    main()



#                precision    recall  f1-score   support

#        akiec       0.25      0.01      0.03        75
#          bcc       0.34      0.25      0.29       105
#          bkl       0.41      0.24      0.30       224
#           df       0.00      0.00      0.00        24
#          mel       0.41      0.33      0.36       233
#           nv       0.77      0.92      0.84      1320
#         vasc       0.68      0.77      0.72        22

#     accuracy                           0.70      2003
#    macro avg       0.41      0.36      0.36      2003
# weighted avg       0.64      0.70      0.65      2003




