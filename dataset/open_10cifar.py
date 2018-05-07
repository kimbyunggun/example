import os.path
import pickle
import os
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) #현재 파일의 path의 dirctory_path를 알려준다.


print(dataset_dir)

def unpickle(file): #load_data from file
    file_path = os.path.join(dataset_dir, file) #불러오려고하는 파일의 path
    print('loading file :',file_path)
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    cifar_data = dict[b'data'] # np_ file shape (10000,32*32)
    cifar_labels = dict[b'labels']

    return cifar_data , cifar_labels

batch1_data , batch1_labels = unpickle(file = 'data_batch_1') #파일 불러오기

print(np.shape(batch1_labels))
print(batch1_labels)
