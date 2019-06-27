import numpy as np
np.seterr(divide='ignore',invalid='ignore')

def compute_ap(query, result):
    """
    :param query:  待查询图像的标记
    :param result:  查询结果的标记
    :return:
    """
    # 查询的总的图片数目
    nimgquery = len(query)
    # 返回结果的图片数
    nres = len(result[0])
    aps = []
    for i,res in enumerate(result):
        res_temp = np.zeros((nres,), dtype=np.float32)
        prec_temp = np.zeros((nres,), dtype=np.float32)
        positive = 0
        for j,e in enumerate(res):
            if query[i] == e:
                res_temp[j] = 1
        res_temp = res_temp.cumsum()
        for j,e in enumerate(res):
            if query[i] == e:
                prec_temp[j]=res_temp[j]/(j+1)
                positive += 1
        if positive == 0:
            ap = 0
        else:
            ap = prec_temp.sum()/positive
        aps.append(ap)
    mAP = np.sum(aps)/nimgquery
    return mAP,aps

if __name__ == "__main__":
    query = [1,1]
    res = np.array([[1,0,1,0,0,1,0,0,1,1],[0,1,0,0,1,0,1,0,0,0]])
    result = compute_ap(query, res)
    print(result)
