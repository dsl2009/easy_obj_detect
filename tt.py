from data_gen import get_batch
from dsl_data import data_loader_multi
import time
from tensorflow.contrib import slim
import tensorflow as tf
tf.enable_eager_execution()

logits = [[0, 0, 0, 0,],[0, 0, 0, 0,],[0, 0, 0.0, 0,],[0, 0, 0, 0,]]

labels = [[0, 0, 0, 0,],[0, 0, 0, 0,],[0, 0, 1.0, 0,],[0, 0, 0, 0,]]
x = slim.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
print(tf.reduce_mean(x))
x = tf.nn.weighted_cross_entropy_with_logits( targets=labels, logits=logits,pos_weight=30)

print(tf.reduce_mean(x))
