import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt 
from tqdm import tqdm

# %matplotlib inline
# Lệnh %matplotlib inline là một magic command trong Jupyter Notebook (và các môi trường tương tự) cho phép 
# bạn hiển thị các biểu đồ và đồ thị được vẽ bằng thư viện Matplotlib trực tiếp trong notebook. 
# Khi bạn sử dụng lệnh này, các biểu đồ sẽ được nhúng ngay dưới ô (cell) mà bạn gọi lệnh vẽ.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms