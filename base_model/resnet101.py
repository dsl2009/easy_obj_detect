from nets.resnet_v2 import resnet_v2_block,resnet_v2,resnet_arg_scope
import tensorflow as tf
from tensorflow.contrib import slim
import config
def resnet_arg_scope_batch_norm(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'trainable':False,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'fused': None,  # Use fused batch norm if possible.
  }
  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def resnet_arg_scope_group_norm(weight_decay=0.0001,
                                activation_fn=tf.nn.relu,
                               ):
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.group_norm,
            ):
        with slim.arg_scope([slim.group_norm]):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc

def second_arg_bn():
    batch_norm_params = {
        'is_training':True,
        'decay': 0.9997,
        'epsilon':1e-5,
        'scale':True
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding = 'SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params) as sc:
            return sc

def second_arg_gn():

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.group_norm,
            padding = 'SAME') as sc:
        return sc




def resnet_v2_101(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_101'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""

  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=2),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
if config.is_use_group_norm:
    base_arg = resnet_arg_scope_group_norm
    second_arg = second_arg_gn
else:
    base_arg = resnet_arg_scope_batch_norm
    second_arg = second_arg_bn
def depwise_cov(x):
    x1 = slim.conv2d(x, 256, kernel_size=[7, 1], stride=1)
    x1 = slim.conv2d(x1, 256, kernel_size=[1, 7], stride=1, activation_fn=None)
    x2 = slim.conv2d(x, 256, kernel_size=[1, 7], stride=1)
    x2 = slim.conv2d(x2, 256, kernel_size=[7, 1], stride=1, activation_fn=None)
    x = x1 + x2
    return x


def fpn_re(img):
    with slim.arg_scope(base_arg()):
        _, endpoint = resnet_v2_101(img)
    c1 = endpoint['resnet_v2_101/block1']
    c2 = endpoint['resnet_v2_101/block2']
    c3 = endpoint['resnet_v2_101/block3']
    c4 = endpoint['resnet_v2_101/block4']

    p3 = slim.conv2d(c3, 256, 1, activation_fn=None)
    p3_upsample = tf.image.resize_bilinear(p3, tf.shape(c2)[1:3])
    p3 = depwise_cov(p3)

    p2 = slim.conv2d(c2, 256, 1, activation_fn=None)
    p2 = p2 + p3_upsample
    p2_upsample = tf.image.resize_bilinear(p2, tf.shape(c1)[1:3])
    p2 = depwise_cov(p2)

    p1 = slim.conv2d(c1, 256, 1, activation_fn=None)
    p1 = p1 + p2_upsample
    p1 = depwise_cov(p1)

    p4 = slim.conv2d(c4, 512,kernel_size=1)
    p4 = depwise_cov(p4)

    return p1,p2,p3,p4



def fpn(img):
    with slim.arg_scope(base_arg()):
        _, endpoint = resnet_v2_101(img)
    c1 = endpoint['resnet_v2_101/block1']
    c2 = endpoint['resnet_v2_101/block2']
    c3 = endpoint['resnet_v2_101/block3']
    c4 = endpoint['resnet_v2_101/block4']
    with slim.arg_scope(second_arg()):
        p5 = slim.conv2d(c3, 256, 1, activation_fn=None)
        p5_upsample = tf.image.resize_bilinear(p5, tf.shape(c2)[1:3])
        p5 = slim.nn.relu(p5)
        p5 = slim.conv2d(p5, 256, 3, rate=2)
        p5 = slim.conv2d(p5, 256, 3, activation_fn=None)

        p4 = slim.conv2d(c2, 256, 1, activation_fn=None)
        p4 = p4 + p5_upsample
        p4_upsample = tf.image.resize_bilinear(p4, tf.shape(c1)[1:3])
        p4 = slim.nn.relu(p4)
        p4 = slim.conv2d(p4, 256, 3, rate=2)
        p4 = slim.conv2d(p4, 256, 3, activation_fn=None)

        p3 = slim.conv2d(c1, 256, 1, activation_fn=None)
        p3 = p3 + p4_upsample
        p3 = slim.nn.relu(p3)
        p3 = slim.conv2d(p3, 256, 3, rate=2)
        p3 = slim.conv2d(p3, 256, 3, activation_fn=None)

        p6 = slim.conv2d(c4,1024,kernel_size=1)
        p6 = slim.conv2d(p6, 512, 3, rate=2)
        p6 = slim.conv2d(p6, 256, kernel_size=3, stride=1, activation_fn=None)

        p7 = slim.nn.relu(p6)
        p7 = slim.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)

        bn = [p3, p4, p5, p6, p7]

        return bn