import tensorflow as tf
import numpy as np
from libs.lib_psalign_pooling import psalign_pooling_op
from libs.lib_psalign_pooling  import psalign_pooling_op_grad
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# pdb.set_trace()

rois = tf.convert_to_tensor([[0, 0, 0, 4, 4], [1, 0, 0, 2, 4], [
                            0, 0, 0, 1, 1]], dtype=tf.float32)
hh = tf.convert_to_tensor(np.random.rand(1, 5, 5, 25*4), dtype=tf.float32)
#hh= tf.transpose(hh, [0, 3, 1, 2])


[y2, channels, argmax_position] = psalign_pooling_op.psalign_pool(hh, rois, group_size=5, sample_height=2, sample_width=2, spatial_scale=1.0)
# [y2, channels] = psalign_pooling_op.psalign_pool(
#     hh, rois, 5, 2, 2, 1.0)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

ny2, nch, nmax = sess.run([y2, channels, argmax_position])
print(ny2)
print(ny2.shape)
print(ny2.shape)
