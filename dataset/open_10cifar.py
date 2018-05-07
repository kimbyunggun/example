import os.path
import pickle
import os
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) #현재 파일의 path의 dirctory_path를 알려준다.


print(dataset_dir)

def unpickle(file):
    file_path = os.path.join(dataset_dir, file) #불러오려고하는 파일의 path
    print('loading file :',file_path)
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return

data = unpickle(file = 'data_batch_1')
