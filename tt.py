
from dsl_data import data_loader_multi
import time
from tensorflow.contrib import slim
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
a = [[1,2,3,4,],[6,6,7,8]]
b = tf.pad(a, [(0, 2), (0, 0)])
print(b)

