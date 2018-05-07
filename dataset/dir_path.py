import os.path
import gzip
import pickle
import os
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__)) #현재 파일의 path의 dirctory_path를 알려준다.

file_dir = os.path.abspath(__file__)  #현재 file의 path를 알려준다.

print(file_dir)
print(dataset_dir)
