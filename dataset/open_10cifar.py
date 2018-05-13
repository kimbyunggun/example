import os.path
import pickle
#pickle.dump() or pickle.load()
import os
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) #현재 파일의 path의 dirctory_path를 알려준다.

print(dataset_dir)

def unpickle(file): #load_data from file 파일로부터 데이터 불러오기
    file_path = os.path.join(dataset_dir, file) #불러오려고하는 파일의 path
    print('loading file :',file_path) # 다음 경로의 파일 불러오기
    with open(file_path, 'rb') as fo: # 파일 불러오기
        dict = pickle.load(fo, encoding='bytes')
    cifar_data = dict[b'data'] # np_ file shape (10000,3*32*32=3072) 10000개의 샘플
    cifar_labels = dict[b'labels'] #np_file shape(,10000) data별 label (0~9)10개

    return cifar_data , cifar_labels

def arrange_data(): # 배치 숫자별 데이터 정리

    batch_data  = {}
    batch_labels = {}
    batch_data['a'] = []
    batch_labels['a'] = []
    batch_data['1'] ,  batch_labels['1']  = unpickle(file = 'data_batch_1')
    batch_data['2'] ,  batch_labels['2']  = unpickle(file = 'data_batch_2')
    batch_data['3'] ,  batch_labels['3']  = unpickle(file = 'data_batch_3')
    batch_data['4'] ,  batch_labels['4']  = unpickle(file = 'data_batch_4')
    batch_data['5'] ,  batch_labels['5']  = unpickle(file = 'data_batch_5')
    batch_data['t'] ,  batch_labels['t']  = unpickle(file = 'test_batch')
    for i in ('1','2','3','4','5'): #x.extend(a) = x라는 리스트에 a라는 리스트를 더함
        batch_data['a'].extend(batch_data[i])
        batch_labels['a'].extend(batch_labels[i])
    batch_data['a'] = np.array(batch_data['a'])
    batch_labels['a'] = np.array(batch_labels['a'])
    return batch_data, batch_labels


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_cifar(i, normalize=True, flatten=True, one_hot_label=False): #i번째 batch
    """MNIST 데이터셋 읽기

    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label :
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다.
    i : batch_datset 고르기 = > 1,2,3,4,5,a

    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)

    """
    dataset = {} #dataset은 batch에 해당하는 파일의 데이터를 불러와 성분별 key로 나눠 정리한 dict
    batch_data , batch_labels = arrange_data()
    dataset['train_img'], dataset['train_label'] = batch_data[i], np.array(batch_labels[i])
    dataset['test_img'], dataset['test_label'] = batch_data['t'], np.array(batch_labels['t'])

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape([-1,3,32,32])
            # dataset[key] = np.rollaxis(dataset[key],1,4)
            # dataset[key] = np.squeeze(dataset[key], axis=0) #(32,32,3)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    batch_data, batch_labels = arrange_data()
    print(batch_data['a'].shape)
