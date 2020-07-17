#!/usr/bin/env python
# coding: utf-8

#DATALOADING
get_ipython().system('wget --quiet -O /resources/data/train_data_keras.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/train_data_keras.tar.gz -C /resources/data --exclude '.*'")

get_ipython().system('wget --quiet -O /resources/data/validation_data_keras.tar.gz https://cocl.us/DL0320EN_VALID_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/validation_data_keras.tar.gz -C /resources/data --exclude '.*'")

get_ipython().system('wget --quiet -O /resources/data/test_data_keras.tar.gz https://cocl.us/DL0320EN_TEST_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/test_data_keras.tar.gz -C /resources/data --exclude '.*'")

import matplotlib.pyplot as plt
from PIL import Image

Input = "/resources/data/train_data_keras/5/0.jpeg"
img = Image.open(Input)
plt.imshow(img)

plt.imshow(Image.open("/resources/data/train_data_keras/200/52.jpeg"))

plt.imshow(Image.open("/resources/data/validation_data_keras/5/0.jpeg"))

plt.imshow(Image.open("/resources/data/validation_data_keras/50/36.jpeg"))

#PREPROCESSING

get_ipython().system('wget --quiet -O /resources/data/train_data_keras.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/train_data_keras.tar.gz -C /resources/data --exclude '.*'")

get_ipython().system('wget --quiet -O /resources/data/validation_data_keras.tar.gz https://cocl.us/DL0320EN_VALID_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/validation_data_keras.tar.gz -C /resources/data --exclude '.*'")


import keras
from keras.preprocessing.image import ImageDataGenerator



import os
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image
import numpy as np 


TARGET_SIZE = (224, 224)
BATCH_SIZE = 5
CLASSES = ['5', '10', '20', '50', '100', '200', '500']
RANDOM_SEED = 0



train_data_dir = '/resources/data/train_data_keras'


train_generator = ImageDataGenerator().flow_from_directory(train_data_dir
                                                           , target_size=TARGET_SIZE
                                                           , batch_size=BATCH_SIZE
                                                           , classes=CLASSES
                                                           , seed=RANDOM_SEED)


validation_data_dir = '/resources/data/validation_data_keras'

valid_generator = ImageDataGenerator().flow_from_directory(validation_data_dir
                                                         ,target_size=TARGET_SIZE
                                                         ,batch_size=BATCH_SIZE
                                                         ,classes=CLASSES
                                                         ,seed=RANDOM_SEED)

batch_1 = train_generator.next()

batch_1 = valid_generator.next()[0]
batch_1 = batch_1.astype(np.uint8)
for i in range(len(batch_1)):
    plt.imshow(batch_1[i])
    plt.show()

#MODEL BUILDING



# Download Training Dataset
get_ipython().system('wget --quiet -O /resources/data/train_data_keras.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/train_data_keras.tar.gz -C /resources/data --exclude '.*'")

# Download Validation Dataset
get_ipython().system('wget --quiet -O /resources/data/validation_data_keras.tar.gz https://cocl.us/DL0320EN_VALID_TAR_KERAS')
get_ipython().system("tar -xzf /resources/data/validation_data_keras.tar.gz -C /resources/data --exclude '.*'")


import keras
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Model


import os
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random


train_data_dir = '/resources/data/train_data_keras'
validation_data_dir = '/resources/data/validation_data_keras'
classes = ['5', '10', '20', '50', '100', '200', '500']

train_generator = ImageDataGenerator().flow_from_directory(train_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=10
                                                           , classes=classes
                                                           , seed=0
                                                           , shuffle=True)


valid_generator = ImageDataGenerator().flow_from_directory(validation_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=5
                                                           , classes=classes
                                                           , seed=0
                                                           , shuffle=True)


base = ResNet50(weights='imagenet')

for layer in base.layers:
    layer.trainable = False

sec_las_base  = base.layers[-2].output

conn_model = Dense(len(classes),activation = 'softmax')(sec_las_base)
base_input =  base.input
model = Model(inputs  = base_input , outputs  = conn_model)


model.compile(optimizer = 'adam',loss="categorical_crossentropy",metrics =['accuracy'])

N_epochs = 20
steps = train_generator. n // train_generator.batch_size
model.fit_generator(generator = train_generator , validation_data = valid_generator , steps_per_epoch = steps , epochs =N_epochs)

train_history = model.history.history
from matplotlib import pyplot as plt

plt.plot(train_history['loss'])
plt.plot(train_history['val_loss'])
plt.title("Model Loss")
plt.ylabel("loss")
plt.xlabel("Epoch")
plt.legend(['train','validation'],loc='upper right')

plt.plot(train_history['acc'])
plt.plot(train_history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

test_valid_generator = ImageDataGenerator().flow_from_directory(validation_data_dir
                                                           , target_size=(224, 224)
                                                           , batch_size=5
                                                           , classes=classes
                                                           , seed=0
                                                           , shuffle=False)


random.seed(0)
numbers = [random.randint(0, 69) for i in range(0, 5)]

# Type your code here
pred = np.argmax(model.predict_generator(test_valid_generator,14)[numbers],axis=1)
class_labels = train_generator.class_indices
pred_labels = [list(class_labels.keys())[list(class_labels.values()).index(i)] for i in pred]
true_labels = [test_valid_generator.filenames[i].split('/')[0] for i in numbers]
ndx = 0
for i in numbers:
    result_string = 'predicted: {}'.format(pred_labels[ndx])
    if pred_labels[ndx] == true_labels[ndx]:
        result_string += ' (Correctly classified)'
    else:
        result_string += ' (Incorrectly classified)'
    plt.imshow(test_valid_generator[i // 5][0][i % 5].astype(np.uint8), aspect='auto')
    plt.show()
    print(result_string)
    ndx += 1

base1 = VGG16(weights = "imagenet")

for layer in base1.layers:
    layer.trainable = False


sec_last_base1 = base1.layers[-2].output
connected_model1=Dense(len(classes),activation = 'softmax')(sec_last_base1)
base1_input = base1.input
model_vgg = Model (input = base1_input,output = connected_model1)
model_vgg.summary()


N_EPOCHS = 5
STEPS = train_generator.n // train_generator.batch_size

model_vgg.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_vgg.fit_generator(generator=train_generator,steps_per_epoch=STEPS, epochs=N_EPOCHS, validation_data=valid_generator)


train_history_vgg = model_vgg.history.history


plt.plot(train_history_vgg['acc'])
plt.plot(train_history_vgg['val_acc'])
plt.title('Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")



plt.plot(train_history_vgg['loss'])
plt.plot(train_history_vgg['val_loss'])
plt.title('Loss')
plt.xlabel("epochs")
plt.ylabel("Loss")


model.save("resnet50_keras.pt")
model_vgg.save("vgg16_keras.pt")

