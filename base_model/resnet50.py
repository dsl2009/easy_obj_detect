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

def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  print(inputs)
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=2),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
if config.is_use_groupnorm:
    base_arg = resnet_arg_scope_group_norm
else:
    base_arg = resnet_arg_scope

if config.is_user_group_norm:
    base_arg = resnet_arg_scope_group_norm
else:
    base_arg = resnet_arg_scope


def fpn(img):
    with slim.arg_scope(base_arg()):
        _, endpoint = resnet_v2_50(img)
    c1 = endpoint['resnet_v2_50/block1']
    c2 = endpoint['resnet_v2_50/block2']
    c3 = endpoint['resnet_v2_50/block3']
    c4 = endpoint['resnet_v2_50/block4']

    p5 = slim.conv2d(c3, 256, 1, activation_fn=None)
    p5_upsample = tf.image.resize_bilinear(p5, tf.shape(c2)[1:3])
    p5 = slim.conv2d(p5, 256, 3, rate=4)
    p5 = slim.conv2d(p5, 256, 3, activation_fn=None)

    p4 = slim.conv2d(c2, 256, 1, activation_fn=None)
    p4 = p4 + p5_upsample
    p4_upsample = tf.image.resize_bilinear(p4, tf.shape(c1)[1:3])
    p4 = slim.conv2d(p4, 256, 3, rate=4)
    p4 = slim.conv2d(p4, 256, 3, activation_fn=None)

    p3 = slim.conv2d(c1, 256, 1, activation_fn=None)
    p3 = p3 + p4_upsample
    p3 = slim.conv2d(p3, 256, 3, rate=4)
    p3 = slim.conv2d(p3, 256, 3, activation_fn=None)

    p6 = slim.conv2d(c4,1024,kernel_size=1)
    p6 = slim.conv2d(p6, 256, 3, rate=4)
    p6 = slim.conv2d(p6, 256, kernel_size=3, stride=1, activation_fn=None)

    p7 = slim.nn.relu(p6)
    p7 = slim.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)

    if config.is_use_last:
        p8 = slim.nn.relu(p7)
        p8 = slim.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)
        return [p3, p4, p5, p6, p7,p8]
    else:
        return [p3, p4, p5, p6, p7]
