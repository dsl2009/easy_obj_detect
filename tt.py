from data_gen import get_batch
from dsl_data import data_loader_multi
import time
from tensorflow.contrib import slim
import np_utils
import tensorflow as tf
tf.enable_eager_execution()
print(np_utils.gen_anchors_single().shape)
print(np_utils.gen_ssd_anchors().shape)
