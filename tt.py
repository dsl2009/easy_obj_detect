from imgaug import augmenters as iaa
import numpy as np
import requests
import json
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from preprocessing import inception_preprocessing

a = tf.constant([[0,1,2,3,4],[1,2,3,4,5],[1,3,3,3,3]],dtype=tf.float32)
b = tf.constant([1,0,1,0],dtype=tf.float32)
c = a[:,0]
d = a[:,1:]
ix = tf.where(tf.equal(c, 1))[:,0]
idx, x, y, x1, y1 = tf.split(a, 5, axis=1)

e = tf.concat([idx, y, x, y1, x1],axis=1)

with tf.Session() as sess:
    print(sess.run(e))
    print(sess.run(idx))
    print(sess.run(y1))
    print(sess.run(tf.gather(d, ix)))




