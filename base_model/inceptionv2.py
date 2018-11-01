from nets import inception_v2,resnet_v2,inception_v3
import tensorflow as tf
from tensorflow.contrib import slim


def inception_v2_ssd(img):
    with slim.arg_scope(inception_v2.inception_v2_arg_scope()):
        logits, end_point = inception_v2.inception_v2_base(img)
    c1 = end_point['Mixed_3c']
    c2 = end_point['Mixed_4e']
    c3 = end_point['Mixed_5c']
    return c1,c2,c3
def fpn(img):
    c1, c2, c3 = inception_v2_ssd(img)
    p5 = slim.conv2d(c3, 256, 1, activation_fn=None)
    p5_upsample = tf.image.resize_bilinear(p5, tf.shape(c2)[1:3])
    p5 = slim.conv2d(p5, 256, 3, activation_fn=None)

    p4 = slim.conv2d(c2, 256, 1, activation_fn=None)
    p4 = p4 + p5_upsample
    p4_upsample = tf.image.resize_bilinear(p4, tf.shape(c1)[1:3])
    p4 = slim.conv2d(p4, 256, 3, activation_fn=None)

    p3 = slim.conv2d(c1, 256, 1, activation_fn=None)
    p3 = p3 + p4_upsample
    p3 = slim.conv2d(p3, 256, 3, activation_fn=None)

    p6 = slim.conv2d(c3, 256, kernel_size=3, stride=2, activation_fn=None)

    p7 = slim.nn.relu(p6)
    p7 = slim.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)

    return [p3, p4, p5, p6, p7]


