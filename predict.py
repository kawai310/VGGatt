import keras
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv2DTranspose
from keras.layers import Multiply, Add, Input, Reshape, Flatten, RepeatVector, Permute

import numpy as np
import pathlib
from PIL import Image
import os
import cv2

###############################################################################################
inputs = Input(shape=(224, 224, 3))
# conv_1
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(model)
model = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(model)

# conv_2
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(model)
model = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(model)

# conv_3
model = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(model)
model = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(model)
model = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(model)
model = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(model)

# conv_4
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(model)

# conv_5
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(model)


# Feature
Fea = model

model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='att1_pool')(model)
model = Conv2D(64, (1, 1), activation='relu', padding='same', name='att1_conv1')(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='att1_conv2')(model)
model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='att2_pool')(model)
model = Conv2D(64, (1, 1), activation='relu', padding='same', name='att2_conv1')(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='att2_conv2')(model)
model = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='att2_conv3')(model)
model = Conv2DTranspose(1, (4, 4), strides=(4, 4), padding='same')(model)

att = Flatten()(model)
att = RepeatVector(512)(att)
att = Permute((2, 1))(att)
att = Reshape((28, 28, 512))(att)
attMap = Multiply()([Fea, att])
salMap = Add()([Fea, attMap])
salMap = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(salMap)
salMap = Conv2DTranspose(1, (4, 4), strides=(4, 4), padding='same')(salMap)
salMap = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same')(salMap)
model = Model(input=inputs,output=salMap)

# Load weights
model.load_weights(os.path.join("./weight_dir", "VGGatt-30.h5"))
###############################################################################################

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])

###############################################################################################
test_list = np.empty([0, 224, 224, 3])
size_list = []
name_list = []
for file in os.listdir('./data/test'):
    test_path = './data/test/' + file
    name_list.append(file)
    test = Image.open(test_path)
    size_list.append(test.size)
    test = np.array(test.resize((224,224)))/255
    test = np.reshape(test, (1, 224, 224, 3))
    test_list = np.concatenate((test_list, test), axis=0)

results = model.predict(test_list, batch_size=1)

###############################################################################################
if not os.path.isdir('./results_dir'):
    os.makedirs('./results_dir')

for i in range(results.shape[0]):
    result = results[i]
    print(np.max(result))
    result = result.reshape([224, 224])
    salMap = cv2.applyColorMap(np.uint8(255*result), cv2.COLORMAP_JET)
    salMap = cv2.cvtColor(salMap, cv2.COLOR_BGR2RGB)
    salMap = Image.fromarray(salMap)
    salMap = salMap.resize(size_list[i])
    salMap.save(os.path.join("./results_dir", name_list[i]))
