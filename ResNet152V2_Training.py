 # Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import matplotlib.pyplot as plt
# %matplotlib inline


import os
os.chdir('/mnt/Documents/Project⁄Thesis/ResNet152V2/tomato')
os.listdir()
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split=0.3)
# Training and validation dataset
train = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/ResNet152V2/tomato/train', seed=123, subset='training')
val = datagen.flow_from_directory('/mnt/Documents/Project⁄Thesis/ResNet152V2/tomato/train', seed=123, subset='validation')

# Test dataset for evaluation
datagen2 = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

test = datagen2.flow_from_directory('/mnt/Documents/Project⁄Thesis/ResNet152V2/tomato/tomato/val')

classes = os.listdir('/mnt/Documents/Project⁄Thesis/ResNet152V2/tomato/tomato/train')


def get_model():
    
    base_model = ResNet152V2(input_shape=(256,256,3), include_top=False)
    
    for layers in base_model.layers[:140]:
        layers.trainable = False
    for layers in base_model.layers[140:]:
        layers.trainable = True
        
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()
    x = Dense(1000, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    
    x = BatchNormalization()(x)
    x = Dense(500, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    
    
    pred = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=pred)
    
    return model

from keras.callbacks import ModelCheckpoint
filepath = "ResNet152V2.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')
callbacks_list = [checkpoint]



model = get_model()
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics='accuracy')
history = model.fit(train, batch_size=80, epochs=50, validation_data=val,callbacks=[callbacks_list,tf.keras.callbacks.CSVLogger('history_resnet152v2.csv')])

model.save('filepath')

history = pd.read_csv('history_resnet152v2.csv') 
history = history['accuracy']
# history = max(history['accuracy'])           # maximum accuracy valuee from the csv file
history


plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train','Test'],loc='best')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train','Test'],loc='best')
plt.show()


model = tf.keras.models.load_model('/mnt/Documents/Project⁄Thesis/ResNet152V2/ResNet152V2.h5')

model.evaluate(test)




