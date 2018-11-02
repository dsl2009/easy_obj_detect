from dsl_data import data_loader_multi
import time
from tensorflow.contrib import slim
from nms import nms
import tensorflow as tf
import numpy as np
import cv2
import torch
from torch.nn import functional as F
tf.enable_eager_execution()





a = tf.ones([3, 5, 4],dtype=tf.float32)
nn = []
for x in range(3):
    b = a[x]
    print(b)
    idx = tf.ones([tf.shape(b)[0],1])*x
    print(idx)
    print(tf.concat([idx, b], axis=1))
    nn.append(tf.concat([idx, b], axis=1))
print(tf.concat(nn,axis=0))

