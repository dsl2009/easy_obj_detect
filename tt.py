import numpy as np
import requests
import json
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from preprocessing import inception_preprocessing
tf.enable_eager_execution()
a = tf.random_uniform(shape=[4, 8])
print(a)
c = [1,2,3,4,5,6,7,8]
d = [1,2,3,4,5,6,7,8]
print(tf.gather(c, tf.where(tf.greater(c, 3))))
print(tf.reduce_sum(tf.cast(tf.greater(c, 2),tf.float32)))
print(tf.stack([c,d],axis=0))
b = ix = tf.nn.top_k(a, 4, sorted=True,
                     name="top_anchors").indices

a1 = tf.ones(shape=[4],dtype=tf.int32)
a2 = tf.ones(shape=[4],dtype=tf.int32)*2
d = tf.stack([a1,a2],axis=1)
print(d)
print(tf.gather_nd(a,d))





