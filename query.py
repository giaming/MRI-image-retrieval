import os
import numpy as np
import h5py
import matplotlib.image as mpimg
from keras.preprocessing import image

from evaluation import compute_ap
from distance_metric import compute_cosin_distance
from extract_features import VGGNet

# the trained features file
indexFile = "featureCNN.h5"
# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(indexFile,'r')
feats = h5f['dataset_1'][:]
# print("features.shape = {}\n".format(feats.shape))
imgNames = h5f['dataset_2'][:]
h5f.close()

model = VGGNet()

# 获得所有的图片
def get_images(database):
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
    print("imgs.shape = {}".format(imgs.shape))
    return imgs



def query(database, queryImage, maxres):
    """
    输入查询图像，从数据库中检索出最相似的图像
    :param database:
    :param queryImg:
    :param maxres:
    :return:
    """
    queryImg = mpimg.imread(queryImage)
    # 提取特征
    queryVec = model.extract_feat(queryImage)

    # print("the shape of queryVec:",np.shape(queryVec))
    # print("the shape of feats:",np.shape(feats))

    scores, rank_ID = compute_cosin_distance(queryVec, feats)   # 使用余弦距离
    # rank_score = scores[rank_ID]
    # print(rank_score)

    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    # print("top %d images in order are: " %maxres, imlist)
    imLabel = []
    for imName in imlist:
        imLabel.append(str(imName, encoding="utf-8").split('_')[2])
    return imLabel




def main():
    # init
    queryDir = "database/test"
    # database file's path
    database = "database/train"
    # number of top retrieved images to show
    maxres = 10
    # imgs = get_images(database)

    resultLabels, queryLabels = [],[]
    for i,f in enumerate(os.listdir(queryDir)):
        queryname_path = os.path.join(queryDir, f)
        queryLabels.append(f.split("_")[2])
        print("Querying the {}th image in Test set.".format(i))
        resultLabel = query(database,queryname_path, maxres)
        resultLabels.append(resultLabel)
    np.savetxt('output/queryLabels.txt',np.array(queryLabels), delimiter=' ', fmt = '%s')
    np.savetxt('output/resultLabels.txt',np.array(resultLabels), delimiter=' ', fmt = '%s')
    mAP,aps = compute_ap(queryLabels,resultLabels)
    print(mAP)
    print(aps)



if __name__ == "__main__":
    main()