import numpy as np
import os
import shutil
from PIL import Image

# マスク画像の色がついている部分を1に変える
def binarymask(mask):
    x = np.zeros((224, 224, 1))
    new_mask = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
    for i in range(224):
        for j in range(224):
            if new_mask[i][j] != 0:
                x[i,j,0]=1
            #elif new_mask[i][j] == 0:
            #    x[i,j,0]=1
    return x


class ImageDataGenerator(object):
    # インスタンス変数を定義
    def __init__(self):
        self.image_list = np.empty([0, 224, 224, 3])
        self.mask_list = np.empty([0, 224, 224, 1])

    # dataを各ndarrayに保存
    def flow_from_directory(self, batch_size):
        while True:
            for file in os.listdir('./data/img'):
                img_path = './data/img/' + file
                mask_path = './data/mask/' + file
                image = Image.open(img_path)
                image = np.array(image.resize((224,224)))/255
                # モノクロ画像は無視
                if image.ndim == 3:
                    image = np.reshape(image, (1, 224, 224, 3))
                    self.image_list = np.concatenate((self.image_list, image), axis=0)

                    mask = Image.open(mask_path)
                    mask = mask.resize((224, 224))
                    mask = np.array(mask)
                    mask = binarymask(mask)
                    mask = np.reshape(mask, (1, 224, 224, 1))
                    self.mask_list = np.concatenate((self.mask_list, mask), axis=0)

                    # 格納された画像枚数がバッチサイズと等しくなったらtrainに渡す
                    if self.image_list.shape[0] == batch_size:
                        inputs = self.image_list
                        targets = self.mask_list
                        # 各numpy配列を初期化
                        self.image_list = np.empty([0, 224, 224, 3])
                        self.mask_list = np.empty([0, 224, 224, 1])

                        yield inputs, targets
