import tensorflow as tf
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Multiply, Add, Input, Reshape, Flatten, RepeatVector, Permute
from keras.utils.data_utils import get_file
from keras.utils import plot_model
import keras.backend as K

from preprocess import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image
import os

batch_size = 16

###############################################################################################
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

inputs = Input(shape=(224, 224, 3))
# conv_1
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
model = BatchNormalization()(model)
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(model)
model = BatchNormalization()(model)
model = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(model)

# conv_2
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(model)
model = BatchNormalization()(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(model)
model = BatchNormalization()(model)
model = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(model)

# conv_3
model = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(model)
model = BatchNormalization()(model)
model = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(model)
model = BatchNormalization()(model)
model = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(model)
model = BatchNormalization()(model)
model = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(model)

# conv_4
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(model)
model = BatchNormalization()(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(model)
model = BatchNormalization()(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(model)
model = BatchNormalization()(model)

# conv_5
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(model)
model = BatchNormalization()(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(model)
model = BatchNormalization()(model)
model = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(model)
model = BatchNormalization()(model)


# Feature
Fea = model

model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='att1_pool')(model)
model = Conv2D(64, (1, 1), activation='relu', padding='same', name='att1_conv1')(model)
model = BatchNormalization()(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='att1_conv2')(model)
model = BatchNormalization()(model)
model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='att2_pool')(model)
model = Conv2D(64, (1, 1), activation='relu', padding='same', name='att2_conv1')(model)
model = BatchNormalization()(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='att2_conv2')(model)
model = BatchNormalization()(model)
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
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='train')
model.load_weights(weights_path, by_name=True)
###############################################################################################
model.summary()


if not os.path.isdir('./graph_dir'):
    os.makedirs('./graph_dir')
plot_model(model, to_file=os.path.join('./graph_dir', 'VGGatt.png'), show_shapes=True)


# conv_4までの重みをフリーズ
for i in range(len(model.layers)):
    print(i, model.layers[i])

for layer in model.layers[:13]:
    layer.trainable = False

###############################################################################################
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# lossの変化がなければ学習率を下げる
rlop = ReduceLROnPlateau(monitor='loss',
                         factor=0.1,
                         patience=3,
                         verbose=1,
                         mode='min',
                         epsilon=0.0001,
                         cooldown=0,
                         min_lr=0.0000001)


###############################################################################################
# バッチサイズごとにデータを渡す
train_datagen = ImageDataGenerator()

history = model.fit_generator(
          generator=train_datagen.flow_from_directory(batch_size),
          steps_per_epoch=int(np.ceil(len(list(pathlib.Path('./data/img/').iterdir())) / batch_size)),
          epochs=50,
          callbacks=[rlop])


# lossの変動グラフを保存
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./history_dir/loss.png')


if not os.path.isdir('./history_dir'):
    os.makedirs('./history_dir')
plot_history(history)

# weightを保存
if not os.path.isdir('./weight_dir'):
    os.makedirs('./weight_dir')
model.save_weights(os.path.join('./weight_dir', 'VGGatt.h5'))
