from pretrainedmodels import dpn68b
from tensorflow.contrib import slim
import tensorflow as tf
from torch import nn
import keras
from nets import resnet_v2
from nets import inception_v3

def dpn_arg_scope(weight_decay=0.0001,
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
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', biases_initializer=None, scope=scope)


def group_conv(inputs, out_puts, kernel_size, padding, stride,  group_num):
    B, H, W, C = inputs.shape.as_list()
    gconv_num = int(out_puts/group_num)
    g_conv =  tf.split(inputs, group_num, axis=3)
    out = []
    for x_data in g_conv:
        out.append(slim.conv2d(x_data, gconv_num, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=None, normalizer_fn=None))
    out = tf.concat(out, axis=3)
    out = slim.batch_norm(out)
    out = tf.nn.relu(out)
    return out


def cat_bn_act(inputs):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, axis=3)
    ip = slim.batch_norm(inputs,activation_fn=tf.nn.relu)
    return ip

def input_block(inputs, num_fea=64):
    ip = conv2d_same(inputs, num_outputs=num_fea, kernel_size=7, stride=2, scope='Input_block')
    ip = slim.max_pool2d(ip, kernel_size=3, stride=2)
    return ip

def dual_path_block(inputs, num_1x1_a, num_3x3_b, num_1x1_c, inc,  groups, block_type = 'normal', b=False, scope=None):
    with tf.variable_scope(scope):
        x_in = tf.concat(inputs, axis=3) if isinstance(inputs, tuple) else inputs
        if block_type is 'proj':
            key_stride = 1
            has_proj = True
        elif block_type is 'down':
            key_stride = 2
            has_proj = True
        else:
            assert block_type is 'normal'
            key_stride = 1
            has_proj = False
        if has_proj:
            x_s = slim.conv2d(x_in, num_outputs=num_1x1_c+2*inc, kernel_size=1, stride=key_stride)
            x_s1 = x_s[:, :, :, :num_1x1_c]
            x_s2 = x_s[:, :, :, num_1x1_c:]
        else:
            x_s1 = inputs[0]
            x_s2 = inputs[1]

        x_in = slim.conv2d(x_in, num_outputs=num_1x1_a, kernel_size=1, stride=1)
        x_in = group_conv(x_in, out_puts=num_3x3_b, kernel_size=3, padding='SAME', stride=key_stride, group_num=groups)
        if b:
            x_in = cat_bn_act(x_in)
            out1 = slim.conv2d(x_in, num_outputs=num_1x1_c, kernel_size=1, biases_initializer=None, normalizer_fn=None, activation_fn=None)
            out2 = slim.conv2d(x_in, num_outputs=inc, kernel_size=1, biases_initializer=None, normalizer_fn=None, activation_fn=None)
        else:
            x_in = slim.conv2d(x_in, num_outputs=num_1x1_c+inc, kernel_size=1, stride=1)
            out1 = x_in[:, :, :, :num_1x1_c]
            out2 = x_in[:, :, :, num_1x1_c:]

        resid = x_s1+out1
        dense = tf.concat([x_s2, out2], axis=3)
        return resid, dense

def DPN(inputs, k_r = 96, groups=32, b=False, k_sec=(3, 4, 20, 3), inc_sec = (16, 32, 64,128 )):
    end_point = {}
    with tf.variable_scope('dpn'):
        with slim.arg_scope(dpn_arg_scope()):
            bw_factor = 1
            net = input_block(inputs)
            # conv2
            bw = 64 * bw_factor
            inc = inc_sec[0]
            r = k_r
            net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups, block_type='proj',
                                  b=b, scope='cov2_proj')
            for i in range(2, k_sec[0] + 1):
                net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups,
                                      block_type='normal', b=b, scope='cov2_norm_'+str(i))
            end_point['conv2'] = tf.concat(net, axis=3)
            # conv3
            bw = 128 * bw_factor
            inc = inc_sec[1]
            r = k_r * 2
            net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups, block_type='down',
                                  b=b, scope='cov3_down')
            for i in range(2, k_sec[1] + 1):
                net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups,
                                      block_type='normal', b=b, scope='cov3_norm_'+str(i))
            end_point['conv3'] = tf.concat(net, axis=3)
            # conv4
            bw = 256 * bw_factor
            inc = inc_sec[2]
            r = k_r * 4
            net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups, block_type='down',
                                  b=b, scope='cov4_down')
            for i in range(2, k_sec[2] + 1):
                net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups,
                                      block_type='normal', b=b, scope='cov4_norm_'+str(i))
            end_point['conv4'] = tf.concat(net, axis=3)
            # conv5
            bw = 512 * bw_factor
            inc = inc_sec[3]
            r = k_r * 8
            net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups, block_type='down',
                                  b=b, scope='cov5_down')
            for i in range(2, k_sec[3] + 1):
                net = dual_path_block(net, num_1x1_a=r, num_3x3_b=r, num_1x1_c=bw, inc=inc, groups=groups,
                                      block_type='normal',
                                      b=b, scope='cov5_norm_'+str(i))
            end_point['conv5'] = tf.concat(net, axis=3)
            net = cat_bn_act(net)
            return net,end_point

def logist(inputs, num_class):
    net,_ = DPN(inputs, k_r = 128, groups=32, b=False, k_sec=([3, 4, 6, 3]), inc_sec = (16, 32, 24, 64 ))
    B, H, W, C = net.shape.as_list()
    net = slim.avg_pool2d(net, kernel_size=(H,W))
    log= slim.flatten(net)
    pred = slim.fully_connected(log, num_outputs=num_class)
    return pred

def depwise_cov(x):
    x1 = slim.conv2d(x, 256, kernel_size=[7, 1], stride=1)
    x1 = slim.conv2d(x1, 256, kernel_size=[1, 7], stride=1, activation_fn=None)
    x2 = slim.conv2d(x, 256, kernel_size=[1, 7], stride=1)
    x2 = slim.conv2d(x2, 256, kernel_size=[7, 1], stride=1, activation_fn=None)
    x = x1 + x2
    return x

def fpn_re(img):
    net, endpoint = DPN(img, k_r=128, groups=32, b=False, k_sec=([3, 4, 6, 3]), inc_sec=(16, 32, 24, 64))
    print(endpoint)

    c1 = endpoint['conv3']
    c2 = endpoint['conv4']
    c3 = endpoint['conv5']

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

    p4 = slim.conv2d(c3, 512,kernel_size=1)
    p4 = slim.conv2d(c3, 256, kernel_size=3,stride=2)
    p4 = depwise_cov(p4)

    return p1,p2,p3,p4
