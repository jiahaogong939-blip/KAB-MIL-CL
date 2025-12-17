import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.decomposition import PCA
import random
import torch
from sklearn import preprocessing
#读取存储在 .mat 文件中的基因组学数据，并通过自定义的 GenomicWSIDataset 类将这些数据加载到 PyTorch 的数据加载器中，以供机器学习模型（比如神经网络）使用。

def resampler(wsi_data, repeat):
    sample_ = [random.choices(list(wsi_data)) for _ in range(repeat)]
    sample = np.squeeze(np.array(sample_))
    return sample

class GenomicWSIDataset(Dataset):
    def __init__(self, PROJECT, MODE, CLASSES):
        


        # get filename of mat (WSI data in mat)
       
        matfiles = []
        for CLASS in CLASSES:
            WSI_PATH = "../postdata/bootstrapping-2t-100-200/mat/%s/%s/%s/" % (PROJECT, MODE, CLASS)
            matfiles_ = [WSI_PATH+file.name for file in os.scandir(WSI_PATH) if file.name.endswith(".mat")]
            matfiles += matfiles_
        
        self.matfiles = matfiles
        self.CLASSES = CLASSES


    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, item):
        matflie = self.matfiles[item]
        mat = scipy.io.loadmat(matflie)
        # wsi_data = mat["cdfeat"]
        wsi_data = mat["bstdfeat"]

        if wsi_data.shape[0] < 200:
            wsi_data = np.vstack((wsi_data, np.zeros((200 - wsi_data.shape[0], wsi_data.shape[1]), dtype="f")))
            # sample = resampler(wsi_data, 200 - wsi_data.shape[0])
            # wsi_data = np.vstack((wsi_data, sample))

        for i in range(len(self.CLASSES)):
            if self.CLASSES[i] in matflie:
                label = i

        split_matfile = matflie.split("/")[-1]
        patientID = split_matfile.split("_")[0]
        #new
        split_matfile = matflie.split("/")[-1]
        patientID = split_matfile.split("_")[0]  # 直接返回字符串（如'TCGA-A6-2685'）
       

        
        return wsi_data, label, patientID, split_matfile


if __name__ == "__main__":
    
    PROJECT = "CRC-DX"
    #PROJECT = "CRC-KR"
    MODE = "TRAIN"
    CLASSES = ["MSS", "MSIMUT"]
    # CLASSES = ["Classical", "Neural", "Proneural", "Mesenchymal"]
    # CLASSES = ["Proneural", "Mesenchymal"]


    dataset = GenomicWSIDataset(PROJECT, MODE, CLASSES)

    

    for i in range(dataset.__len__()):
        wsi_data, label, patientID, mat_file = dataset.__getitem__(i)
        print("Get %s with %s, wsi shape(%d, %d), mat file %s, wsi_datatype: %s"
              % (patientID, CLASSES[label], wsi_data.shape[0], wsi_data.shape[1], mat_file, wsi_data.dtype))


""" 
import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.decomposition import PCA
import random
import torch
from sklearn import preprocessing

def resampler(wsi_data, repeat):
    sample_ = [random.choices(list(wsi_data)) for _ in range(repeat)]
    sample = np.squeeze(np.array(sample_))
    return sample

class GenomicWSIDataset(Dataset):
    def __init__(self, PROJECT, MODE, CLASSES):
        # 获取.mat文件名（WSI数据在.mat中）
        matfiles = []
        for CLASS in CLASSES:
            WSI_PATH = "../postdata/bootstrapping-2t-100-200/mat/%s/%s/%s/" % (PROJECT, MODE, CLASS)
            matfiles_ = [WSI_PATH + file.name for file in os.scandir(WSI_PATH) if file.name.endswith(".mat")]
            matfiles += matfiles_
        
        self.matfiles = matfiles
        self.CLASSES = CLASSES


    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, item):
        mat_file = self.matfiles[item]
        mat = scipy.io.loadmat(mat_file)
        wsi_data = mat["bstdfeat"]

        # 生成边缘索引
        if wsi_data.shape[0] < 200:
            original_rows = wsi_data.shape[0]
            mask = np.zeros(200, dtype=int)
            mask[:original_rows] = 1
            # 用零填充到200行
            padding = np.zeros((200 - original_rows, wsi_data.shape[1]), dtype=np.float32)
            wsi_data = np.vstack((wsi_data, padding))
        else:
            mask = np.ones(200, dtype=int)
        
        # 获取标签
        for i in range(len(self.CLASSES)):
            if self.CLASSES[i] in mat_file:
                label = i

        # 获取patientID和mat_file信息
        split_matfile = mat_file.split("/")[-1]
        patientID = split_matfile.split("_")[0]

        return wsi_data, label, patientID, mat_file, mask  # 添加mask到返回值

if __name__ == "__main__":
    PROJECT = "CRC-DX"
    # PROJECT = "CRC-KR"
    MODE = "TRAIN"
    CLASSES = ["MSS", "MSIMUT"]
    # CLASSES = ["Classical", "Neural", "Proneural", "Mesenchymal"]
    # CLASSES = ["Proneural", "Mesenchymal"]

    dataset = GenomicWSIDataset(PROJECT, MODE, CLASSES)

    for i in range(dataset.__len__()):
        wsi_data, label, patientID, mat_file, mask = dataset.__getitem__(i)  # 接收mask
        print("Get %s with %s, wsi shape(%d, %d), mat file %s, wsi_datatype: %s"
              % (patientID, CLASSES[label], wsi_data.shape[0], wsi_data.shape[1], mat_file, wsi_data.dtype))
        #print("Edge mask:", mask)  # 打印边缘索引
 """