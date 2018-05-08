import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.open_10cifar import load_cifar
from common.functions import signoid, softmax
