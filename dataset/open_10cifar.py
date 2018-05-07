import os.path
import pickle
import os
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) #현재 파일의 path의 dirctory_path를 알려준다.


print(dataset_dir)

def unpickle(file): #load_data from file
    file_path = os.path.join(dataset_dir, file) #불러오려고하는 파일의 path
    print('loading file :',file_path) # 다음 경로의 파일 불러오기
    with open(file_path, 'rb') as fo: # 파일 불러오기
        dict = pickle.load(fo, encoding='bytes')
    cifar_data = dict[b'data'] # np_ file shape (10000,32*32=3072) 10000개의 샘플
    cifar_labels = dict[b'labels'] #np_file shape(,10000) data별 label (0~9)10개

    return cifar_data , cifar_labels

batch1_data , batch1_labels = unpickle(file = 'data_batch_1') #파일 불러오기

print(np.shape(batch1_labels))
print(batch1_labels)
