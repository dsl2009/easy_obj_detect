# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from utils import faster_rcnn_utils as utils
import cv2
import numpy as np
from nets import inception_v2
import glob
import time
from faster_rcnn_config import config_instace as cfg
from base_model import resnet50
from losses import rcnn_losses as losses
import config
from libs.lib_psalign_pooling import psalign_pooling_op, psalign_pooling_op_grad
def rpn_graph(feature_map, anchors_per_location=config.aspect_num[0]):
    shared = slim.conv2d(feature_map, 512, 3, activation_fn=slim.nn.relu)
    x = slim.conv2d(shared, 2 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_class_logits = tf.reshape(x, shape=[tf.shape(x)[0], -1, 2])
    rpn_probs = slim.nn.softmax(rpn_class_logits)
    print(rpn_probs)
    x = slim.conv2d(shared, 4 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_bbox = tf.reshape(x, shape=[tf.shape(x)[0], -1, 4])
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def rpn_arges():
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            trainable=True,
            padding = 'SAME') as sc:
        return sc


def rpn_net(fpn,num_anchors):
    with slim.arg_scope(rpn_arges()):
        rpn = slim.conv2d(fpn, 512, 3, scope="rpn_conv/3x3")

        rpn_cls_score = slim.conv2d(rpn, num_anchors * 2, 1,  padding='VALID',activation_fn=None, scope='rpn_cls_score')
        rpn_bbox_pred = slim.conv2d(rpn, num_anchors * 4, 1, padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        rpn_class_logits = tf.reshape(rpn_cls_score, shape=[tf.shape(rpn)[0], -1, 2])
        rpn_probs = slim.nn.softmax(rpn_class_logits)
        rpn_bbox = tf.reshape(rpn_bbox_pred, shape=[tf.shape(rpn)[0], -1, 4])
    return rpn_class_logits, rpn_probs, rpn_bbox


def global_context_module(bottom, prefix='', ks=15, chl_mid=256, chl_out=1024):
    with slim.arg_scope( [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=None,
            weights_initializer =tf.random_normal_initializer(mean=0.0, stddev=0.01),
            trainable=True,
            padding = 'SAME'):
        col_max = slim.conv2d(bottom, chl_mid, [ks, 1],scope=prefix + '_conv%d_w_pre' % ks)
        col = slim.conv2d(col_max, chl_out, [1, ks],scope=prefix + '_conv%d_w' % ks)
        row_max = slim.conv2d(bottom, chl_mid, [1, ks], scope=prefix + '_conv%d_h_pre' % ks)
        row = slim.conv2d(row_max, chl_out, [ks, 1],scope=prefix + '_conv%d_h' % ks)
        s = row + col
        return s



def get_rpns(fp):
    rpn_c_l = []
    r_p = []
    r_b = []
    for f in fp:
        rpn_class_logits, rpn_probs, rpn_bbox = rpn_graph(f)
        rpn_c_l.append(rpn_class_logits)
        r_p.append(rpn_probs)
        r_b.append(rpn_bbox)
    rpn_class_logits = tf.concat(rpn_c_l, axis=1)
    rpn_probs = tf.concat(r_p, axis=1)
    rpn_bbox = tf.concat(r_b, axis=1)
    return rpn_class_logits, rpn_probs, rpn_bbox


def covert(rois):
    nn = []
    for x in range(config.batch_size):
        b = rois[x]
        idx = tf.ones([tf.shape(b)[0], 1]) * x
        nn.append(tf.concat([idx, b], axis=1))
    return tf.concat(nn, axis=0)

def propsal(rpn_probs, rpn_bbox):
    scores = rpn_probs[:, :, 1]
    deltas = rpn_bbox
    deltas = deltas * np.reshape(cfg.RPN_BBOX_STD_DEV, [1, 1, 4])
    anchors = tf.constant(cfg.anchors,dtype=tf.float32)
    ix = tf.nn.top_k(scores, 2000, sorted=True,name="top_anchors").indices
    window = np.array([0, 0, 1, 1], dtype=np.float32)
    result = []
    for b in range(config.batch_size):

        scores_tp = tf.gather(scores[b, :], ix[b, :])
        deltas_tp = tf.gather(deltas[b, :], ix[b, :])
        pre_nms_anchors_tp = tf.gather(anchors, ix[b, :])

        boxes_tp = utils.apply_box_deltas_graph(pre_nms_anchors_tp, deltas_tp)
        boxes_tp = utils.clip_boxes_graph(pre_nms_anchors_tp, window)

        props = utils.nms(boxes_tp, scores_tp, cfg)
        result.append(props)
    pro = tf.stack(result, axis=0)
    pro.set_shape([config.batch_size, cfg.NMS_ROIS_TRAINING, 4])
    return pro

def detection_target(input_proposals, input_gt_class_ids, input_gt_boxes):
    roiss = []
    roi_gt_class_idss = []
    deltass = []
    for b in range(config.batch_size):
        proposals = input_proposals[b, :, :]
        gt_class_ids = input_gt_class_ids[b, :]
        gt_boxes = input_gt_boxes[b, :, :]

        proposals, _ = utils.trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = utils.trim_zeros(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

        overlaps = utils.overlaps_graph(proposals, gt_boxes)

        best_true = tf.argmax(overlaps, axis=0)

        roi_iou_max = tf.reduce_max(overlaps, axis=1)

        positive_roi_bool = (roi_iou_max >= 0.7)
        positive_indices = tf.where(positive_roi_bool)[:, 0]

        #positive_indices = tf.concat([best_true, positive_indices], axis=0)
        negative_indices = tf.where(roi_iou_max < 0.3)[:, 0]
        #TRAIN_ROIS_PER_IMAGE =200 ROI_POSITIVE_RATIO=0.33

        positive_count = int(cfg.TRAIN_ROIS_PER_IMAGE * cfg.ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]

        r = 1.0 / cfg.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # 将ROI对应到true_box 和true_label
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # 将对应的ROI 与true_box 进行编码
        deltas = utils.box_encode(positive_rois, roi_gt_boxes)
        deltas /= cfg.BBOX_STD_DEV

        #roi_gt_boxes
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        idx = tf.ones([tf.shape(rois)[0], 1]) * b
        rois = tf.concat([idx, rois], axis=1)

        N = tf.shape(negative_rois)[0]
        #P = tf.maximum(cfg.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        #rois = tf.pad(rois, [(0, P), (0, 0)])
        #roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N)])
        deltas = tf.pad(deltas, [(0, N), (0, 0)])
        roiss.append(rois)
        roi_gt_class_idss.append(roi_gt_class_ids)
        deltass.append(deltas)
    print(tf.concat(roiss, axis=0), tf.concat(roi_gt_class_idss, axis=0), tf.concat(deltass, axis=0))
    return tf.concat(roiss, axis=0), tf.concat(roi_gt_class_idss, axis=0), tf.concat(deltass, axis=0)
    #return tf.stack(roiss, axis=0), tf.stack(roi_gt_class_idss, axis=0), tf.stack(deltass, axis=0)

def roi_poling(rois, fpns):

    bx_ind = rois[:,0]
    box = rois[:,1:]
    result = tf.image.crop_and_resize(fpns, box, box_ind=bx_ind, crop_size=7)
    return result

def ps_roi_polling(rois, ps_fm):
    idx, ymin, xmin, ymax, xmax = tf.split(rois, 5, axis=1)
    rois = tf.concat([idx, xmin*config.image_size[0], ymin*config.image_size[0],
                      xmax*config.image_size[0], ymax*config.image_size[0]], axis=1)


    [psroipooled_rois, _, _] = psalign_pooling_op.psalign_pool(
        ps_fm, rois, group_size=7,
        sample_height=2, sample_width=2, spatial_scale=1.0 / 16.0)
    return psroipooled_rois


def ps_roi_allign(rois, net5, num_cls = 2):

    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    #rois = covert(rois)
    ps_chl = 7*7*10
    ps_fm = global_context_module(
        net5, prefix='conv_new_1',
        ks=15, chl_mid=256, chl_out=ps_chl)
    ps_fm = tf.nn.relu(ps_fm)
    psroipooled_rois = ps_roi_polling(rois=rois, ps_fm=ps_fm)
    psroipooled_rois = slim.flatten(psroipooled_rois)

    ps_fc_1 = slim.fully_connected(
        psroipooled_rois, 2048, weights_initializer=initializer,
        activation_fn=tf.nn.relu, trainable=True, scope='ps_fc_1')

    cls_score = slim.fully_connected(
        ps_fc_1, num_cls, weights_initializer=initializer,
        activation_fn=None, trainable=True, scope='cls_fc')
    bbox_pred = slim.fully_connected(
        ps_fc_1, 4 * num_cls, weights_initializer=initializer_bbox,
        activation_fn=None, trainable=True, scope='bbox_fc')
    cls_prob = tf.nn.softmax(cls_score)
    return cls_prob, cls_score,  bbox_pred


def get_train_tensor(images, input_rpn_match,input_rpn_bbox, gt_label, gt_boxs):
    net4, net5 = resnet50.light_head(images)

    rpn_class_logits, rpn_probs, rpn_bbox = rpn_net(net4, num_anchors=config.aspect_num[0])

    propsal_box = propsal(rpn_probs, rpn_bbox)
    rois, target_class_ids, target_deltas = detection_target(propsal_box, gt_label, gt_boxs)

    tf.summary.scalar('roi',tf.shape(rois)[0])

    cls_prob, cls_score, bbox_pred = ps_roi_allign(rois, net5)


    rpn_class_loss = losses.rpn_class_loss_graph(input_rpn_match, rpn_class_logits)
    rpn_bbox_loss = losses.rpn_bbox_loss_graph(input_rpn_bbox, input_rpn_match, rpn_bbox)

    class_loss = losses.mrcnn_class_loss_graph(target_class_ids, cls_score, bbox_pred)
    bbox_loss = losses.mrcnn_bbox_loss_graph(target_deltas, target_class_ids, bbox_pred)

    tf.losses.add_loss(rpn_class_loss)
    tf.losses.add_loss(rpn_bbox_loss)
    tf.losses.add_loss(class_loss)
    tf.losses.add_loss(bbox_loss)
    total_loss = tf.losses.get_losses()
    tf.summary.scalar(name='rpn_class_loss', tensor=rpn_class_loss)
    tf.summary.scalar(name='rpn_bbox_loss', tensor=rpn_bbox_loss)
    tf.summary.scalar(name='class_loss', tensor=class_loss)
    tf.summary.scalar(name='bbox_loss', tensor=bbox_loss)
    ls = tf.identity(total_loss, 'ss')
    train_tensors = tf.losses.get_total_loss()
    return train_tensors,ls, cls_score


def predict1(images, window):
    net4, net5 = resnet50.light_head(images)
    rpn_class_logits, rpn_probs, rpn_bbox = rpn_net(net4, num_anchors=config.aspect_num[0])
    rois  = propsal(rpn_probs, rpn_bbox)
    propsal_box = tf.squeeze(rois, 0)
    idx = tf.ones([tf.shape(propsal_box)[0], 1]) * 0
    propsal_box = tf.concat([idx, propsal_box], axis=1)


    cls_prob, cls_score, bbox_pred = ps_roi_allign(propsal_box, net5)
    bbox_pred = tf.reshape(bbox_pred, shape=(-1,2,4))


    detections = utils.refine_detections_graph(rois, cls_prob, bbox_pred, window, cfg)

    return detections, propsal_box
def predict(images, window):
    net4, net5 = resnet50.light_head(images)

    rpn_class_logits, rpn_probs, rpn_bbox = rpn_net(net4, num_anchors=config.aspect_num[0])
    print(rpn_class_logits)
    propsal_box = propsal(rpn_probs, rpn_bbox)



    propsal_box = tf.squeeze(propsal_box,0)
    pp = rpn_class_logits[:, 1:]
    cls = tf.argmax(pp, axis=1)
    pp = tf.reduce_max(pp, axis=1)
    ix = tf.where(tf.greater(pp, 0.5))[:, 0]

    score = tf.gather(pp, ix)
    box = tf.gather(propsal_box, ix)
    cls = tf.gather(cls, ix)

    box = tf.clip_by_value(box, clip_value_min=0.0, clip_value_max=1.0)

    keep = tf.image.non_max_suppression(
        scores=score,
        boxes=box,
        iou_threshold=0.2,
        max_output_size=100
    )
    b1 = tf.concat([box[:, 1:2], box[:, 0:1], box[:, 3:], box[:, 2:3]], axis=1)


    return tf.gather(b1, keep), tf.gather(score, keep), tf.gather(cls, keep)






