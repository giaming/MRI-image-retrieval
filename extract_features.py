import numpy as np
from numpy import linalg as LA
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Reshape, Flatten, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


class VGGNet:
    def __init__(self):
        self.weight = 'imagenet'
        model_vgg = VGG16(weights='imagenet', input_shape=(224,224,3), include_top = True)
        input_img = model_vgg.input
        model = model_vgg.get_layer("block3_conv1").output
        model = GlobalAveragePooling2D(name='gap')(model)
        self.model = Model(inputs=input_img, outputs=model,name='Model')
        self.model.summary()

    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])  # 做正则化
        return norm_feat



if __name__ == "__main__":
    net = VGGNet()
    img_path = "database/train/imagedata1_label_0_pid_100360_orientation_0.tif"
    feat = net.extract_feat(img_path)
    print(feat)
    print(np.shape(feat))