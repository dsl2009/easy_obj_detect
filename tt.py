from imgaug import augmenters as iaa
import numpy as np
import requests
import json
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
tf.enable_eager_execution()
a = tf.zeros(shape=[2, 512,512,3])

x = np.asarray([123.15, 115.90, 103.06])/255.0

a = a+x


print(a)
