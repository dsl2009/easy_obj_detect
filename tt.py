from dsl_data import data_loader_multi
import time
from tensorflow.contrib import slim
from nms import nms
import tensorflow as tf
import numpy as np
import cv2
import torch
from torch.nn import functional as F
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss
tf.enable_eager_execution()
a = np.asarray([[1,2,3,4.0],[4,5,6,8.0]],np.float32)
b = np.asarray([[2,2,3,5.0],[1,2,7,9.0]],np.float32)
a1 = torch.from_numpy(a)
b1 = torch.from_numpy(b)
c1 = F.smooth_l1_loss(a1,b1,False)
print(c1)
print(smooth_l1_loss(a,b))
