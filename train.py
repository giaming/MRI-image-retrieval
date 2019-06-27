import os
import h5py
import numpy as np
import csv
import codecs

from extract_features import VGGNet

'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif')]

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+','utf-8')
    writer = csv.writer(file_csv, delimiter=' ',quotechar=' ',quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("Successfully saved.")


if __name__ == "__main__":
    """
    生成图像特征数据库
    """
    db = "database/train"
    img_list = get_imlist(db)
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")
    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
    feats = np.array(feats)
    output = "featureCNN.h5"
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()
