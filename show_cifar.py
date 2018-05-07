# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.open_10cifar import load_cifar
from PIL import Image
import matplotlib.pyplot as plt

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img),mode='RGB')
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_cifar(i = 1, flatten=True, normalize=False)

img = x_train[1]
label = t_train[1]
print(label)

print(img.shape)  # (,3072)
img = img.reshape([-1,3,32,32])  # 형상을 원래 이미지의 크기로 변형
img = np.rollaxis(img,1,4)


print(img.shape)  # (1,32, 32,3)
print(np.uint8(img))
img = np.squeeze(img, axis=0) #axis = 0에 해당하는 dim을 없애준다.
print(np.uint8(img))
print(np.uint8(img).shape)
img_show(img)
