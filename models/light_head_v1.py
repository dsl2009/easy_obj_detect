import tensorflow as tf
from nets import resnet_v1, resnet_utils
from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, initializers, layers
import litht_head_config as cfg
from base_model import resnet50
from utils import light_head_utils


def model(image,is_training):
    num_anchors = len(cfg.anchor_scales) * len(cfg.anchor_ratios)

    net_conv4,net_conv5 = resnet50.light_head(image)
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    with tf.variable_scope(
            'resnet_v1_101', 'resnet_v1_101',
            regularizer=tf.contrib.layers.l2_regularizer(
                cfg.weight_decay)):
        # rpn
        rpn = slim.conv2d(
            net_conv4, 512, [3, 3], trainable=is_training,
            weights_initializer=initializer, activation_fn=nn_ops.relu,
            scope="rpn_conv/3x3")
        rpn_cls_score = slim.conv2d(
            rpn, num_anchors * 2, [1, 1], trainable=is_training,
            weights_initializer=initializer, padding='VALID',
            activation_fn=None, scope='rpn_cls_score')
        rpn_bbox_pred = slim.conv2d(
            rpn, num_anchors * 4, [1, 1], trainable=is_training,
            weights_initializer=initializer, padding='VALID',
            activation_fn=None, scope='rpn_bbox_pred')

        # generate anchor
        height = tf.cast(tf.shape(rpn)[1], tf.float32)
        width = tf.cast(tf.shape(rpn)[2], tf.float32)
        anchors = light_head_utils.generate_anchors_opr(
            height, width, cfg.stride[0], cfg.anchor_scales,
            cfg.anchor_ratios)
        # change it so that the score has 2 as its channel size
        rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
        rpn_cls_prob = tf.nn.softmax(rpn_cls_prob, name='rpn_cls_prob')
        rpn_cls_prob = tf.reshape(rpn_cls_prob, tf.shape(rpn_cls_score))

        rois, roi_scores = light_head_utils.proposal_opr(
            rpn_cls_prob, rpn_bbox_pred, cfg.stride,
            anchors, num_anchors)
