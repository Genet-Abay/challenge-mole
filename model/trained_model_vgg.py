import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam

# Load the dataset
images_dir = r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_images"
df_original = pd.read_csv(r"C:\BeCode\computervisionData\HAM10000_skin_mnist\HAM10000_metadata.csv")
df = df_original.sample(n=100)

# Preprocess the data
df['path'] = df['image_id'].map(lambda x: os.path.join(images_dir, x + '.jpg'))
df['cell_type'] = df['dx'].map({'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6})


# Split the data into training and validation sets
df_train = df.sample(frac=0.8, random_state=0)
df_val = df.drop(df_train.index)
df_train['cell_type'] = df_train['cell_type'].astype(str)
df_val['cell_type'] = df_val['cell_type'].astype(str)
# Create a data generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen.flow_from_dataframe(df_train, x_col='path', y_col='cell_type', target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = datagen.flow_from_dataframe(df_val, x_col='path', y_col='cell_type', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Add a global average pooling layer and a dense layer with 7 units (one for each class)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(7, activation='softmax')(x)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit_generator(train_gen, epochs=5, validation_data=val_gen)

# Plot the training and validation accuracy
sns.set()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()