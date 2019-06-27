import os
import numpy as np
import h5py
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from keras.preprocessing import image

from distance_metric import compute_cosin_distance
from extract_features import VGGNet
from tSNE import plot_tsne
from window import Ui_Form


# the trained features file
indexFile = "featureCNN.h5"
h5f = h5py.File(indexFile,'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()


# database file's path
database = "../database/train/"
# number of top retrieved images to show


# 获得所有的图片
imgs = []
for f in os.listdir(database):
    filename = os.path.splitext(f)  # filename in directory
    filename_full = os.path.join(database,f)  # full path filename
    head, ext = filename[0], filename[1]
    if ext.lower() not in [".tif"]:
        continue
    # Read image file
    img = image.load_img(filename_full, target_size=(224,224))  # resize images as required by the pre-trained model
    imgs.append(np.array(img))  # iamge
imgs = np.array(imgs)

maxres = 16
# queryImgName = ""

class MainWindow(QMainWindow, Ui_Form):
    queryImgPath = ""
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.signalSlots()

    # button与具体方法关联
    def signalSlots(self):
        # 文件按钮相关方法
        self.load_image_btn.clicked.connect(lambda : Openimage(self))
        self.retrieval_btn.clicked.connect(lambda : Queryimage(self))
        self.plot_btn.clicked.connect(lambda : plot_tSNE())

def Openimage(windows):
    queryImgName,_ = QFileDialog.getOpenFileName(None,
                                           r'打开图像',
                                           r'G:\bishe\image_retrieval_cnn\versionVGG16\database',
                                           r'Image Files(*.jpg *.tif *.png)')
    # 利用qlabel显示图片
    png = QtGui.QPixmap(queryImgName).scaled(224,224)
    windows.label_query.setPixmap(png)
    # windows.label_query.setFrameShadow(QtWidgets.QFrame.Raised)
    # windows.label_query.setLineWidth(2)
    windows.queryImgPath = queryImgName


def Queryimage(windows):
    # read query image
    p, f = os.path.split(windows.queryImgPath)
    queryImLabel = f.split('_')[2]
    model = VGGNet()
    queryVec = model.extract_feat(windows.queryImgPath)
    scores, rank_ID = compute_cosin_distance(queryVec, feats)   # 使用余弦距离
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    print("top %d images in order are: " %maxres, imlist)
    imLabel = []
    for imName in imlist:
        imLabel.append(str(imName, encoding="utf-8").split('_')[2])
    print(imLabel)

    for i,f in enumerate(imlist):
        res = QtGui.QPixmap(database+str(f,encoding='utf-8')).scaled(200, 200)
        syntax = "windows.label_res{}.setPixmap(res)".format(i+1)
        eval(syntax)

        syntax1 = "windows.label_res{}.setStyleSheet('background-color: rgb(0, 255, 0)')".format(i+1)
        syntax2 = "windows.label_res{}.setStyleSheet('background-color: rgb(255, 0, 0)')".format(i+1)
        if queryImLabel == imLabel[i]:
            eval(syntax1)
        else:
            eval(syntax2)

        syntax4 = "windows.label_res{}.setAlignment(QtCore.Qt.AlignCenter)".format(i+1)
        eval(syntax4)
        # 脑膜瘤0，胶质瘤1，垂体瘤2
        if int(imLabel[i]) == 0:
            syntax5 = "windows.label{}.setText('脑膜瘤')".format(i+1)
        elif int(imLabel[i]) == 1:
            syntax5 = "windows.label{}.setText('胶质瘤')".format(i + 1)
        else:
            syntax5 = "windows.label{}.setText('垂体瘤')".format(i + 1)
        eval(syntax5)

def plot_tSNE():
    # plot tSNE
    output_tsne_dir = os.path.join("output")
    if not os.path.exists(output_tsne_dir):
        os.makedirs(output_tsne_dir)
    tsne_filename = os.path.join(output_tsne_dir, "tsne_new.png")
    print("Plotting tSNE_new to {}...".format(tsne_filename))
    plot_tsne(imgs, feats, tsne_filename)


if __name__ == '__main__':
    app=QApplication(sys.argv)
    mw=MainWindow()
    mw.show()
    sys.exit(app.exec_())