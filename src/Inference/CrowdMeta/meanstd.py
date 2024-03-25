import cv2
import numpy as np

# 特征提取模块
def feature_extract(image_path):

    img = cv2.imread(image_path).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 通过img.copy()方法，复制img的数据到mean_img
    mean_img = img.copy()
    # 使用 .mean() 方法可得出 mean_img 的平均值
    mean = mean_img.mean()

    # mean_img -= mean_img.mean() 等效于 mean_img = mean_img - mean_img.mean()
    # 减去平均值，得出零平均值矩阵
    mean_img -= mean_img.mean()

    std_img = mean_img.copy()
    # 输出 std_img 的标准差
    std = std_img.std()

    return mean,std


#   数据存储
def batch_extractor(root_path,path):
    with open(root_path, 'r', encoding='utf-8') as fp:
        files = fp.readlines()
    for line in files:
        name = line.split()
    result = []
    for f in name:
        print('Extracting features from image %s' % f)
        mean,std = feature_extract(f)
        result.append([mean,std])

    fp=open(path,'w')
    for j in range(len(result)):
        fp.write(' '.join('%s'% k for k in result[j])+'\n')
    # print(result)
    return result

# 主程序
def run():

    root_path = sys.argv[1]
    path = sys.argv[2]

    batch_extractor(root_path,path)

run()
