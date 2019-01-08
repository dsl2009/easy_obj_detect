# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import slim
from utils import faster_rcnn_utils as utils
import cv2
import numpy as np
import glob
import time
from faster_rcnn_config import config_instace as cfg
from base_model import resnet50, inceptionv2
from utils import np_utils
from losses import loss
import config
def rpn_graph(feature_map, anchors_per_location=config.aspect_num[0]):
    shared = slim.conv2d(feature_map, 512, 3, activation_fn=slim.nn.relu)
    x = slim.conv2d(shared, 2 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_class_logits = tf.reshape(x, shape=[tf.shape(x)[0], -1, 2])
    rpn_probs = slim.nn.softmax(rpn_class_logits)

    x = slim.conv2d(shared, 4 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_bbox = tf.reshape(x, shape=[tf.shape(x)[0], -1, 4])
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def classfy_model(feature_map,ix=0, num_anchors=9):
    with tf.variable_scope('classfy'+str(ix),reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d],activation_fn=slim.nn.relu):
            feature_map = slim.repeat(feature_map,4,slim.conv2d,num_outputs=256,kernel_size=3,stride=1,scope='classfy_repeat')
        out_puts = slim.conv2d(feature_map, config.Config['num_classes'] * num_anchors, kernel_size=3, stride=1,scope='classfy_conv',
                               weights_initializer=tf.initializers.zeros,activation_fn=None)
        out_puts = tf.reshape(out_puts,shape=(config.batch_size,-1, config.Config['num_classes']))

    return out_puts

def regression_model(feature_map,ix=0, num_anchors=9):
    with tf.variable_scope('regression'+str(ix), reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d], activation_fn=slim.nn.relu):
            feature_map = slim.repeat(feature_map, 2, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
        out_puts = slim.conv2d(feature_map, 4 * num_anchors, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        out_puts = tf.reshape(out_puts, shape=(config.batch_size,-1, 4))

    return out_puts



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


def get_odm(fps):
    logits = []
    boxes = []
    for ix, fp in enumerate(fps):
        logits.append(classfy_model(fp, 0, config.aspect_num[ix]))
        boxes.append(regression_model(fp, 0, config.aspect_num[ix]))
    logits = tf.concat(logits, axis=1)
    boxes = tf.concat(boxes, axis=1)
    return logits, boxes

def decode_box(anchors,pred_loc,variance=None):
    if variance is None:
        variance =[0.1, 0.2]
    boxes = tf.concat((
        anchors[:, :2] + pred_loc[:, :2] * variance[0] * anchors[:, 2:],
        anchors[:, 2:] * tf.exp(pred_loc[:, 2:] * variance[1])), 1)
    return  boxes

def encode_box(true_box,anchors,variance=None):
    '''
    :param true_box: [center_x, center_y, w, h]: 
    :param anchors: [center_x, center_y, w, h]: 
    :param variance: 
    :return: 
    '''

    if variance is None:
        variance =[0.1, 0.2]
    g_cxcy = (true_box[:, :2]- anchors[:, :2])/(variance[0] * anchors[:, 2:])
    g_wh = tf.log(true_box[:,2:]/ anchors[:, 2:])/variance[1]

    return tf.concat([g_cxcy, g_wh],axis=1)



def refine_box(pred_rpn_probs, pred_rpn_bbox_offset, input_rpn_class_ids, input_box_offset):
    print(pred_rpn_bbox_offset.shape)
    input_box_offset = tf.cast(input_box_offset, tf.float32)
    anc = config.anchor_gen(config.image_size)
    anchors = tf.constant(anc, dtype=tf.float32)
    box_offset = []
    for b in range(config.batch_size):

        true_boxes = decode_box(anchors, input_box_offset[b])
        pred_boxes = decode_box(anchors, pred_rpn_bbox_offset[b])
        new_box_offset = encode_box(true_boxes, pred_boxes)
        box_offset.append(new_box_offset)

    box_offset = tf.stack(box_offset)

    return box_offset,input_rpn_class_ids















def propsal(rpn_probs, rpn_bbox, input_gt_class_ids, input_gt_boxes):
    scores = rpn_probs[:, :, 1]

    box_deltas = rpn_bbox
    box_deltas = box_deltas * np.reshape(cfg.RPN_BBOX_STD_DEV, [1, 1, 4])
    anchors = tf.constant(cfg.anchors,dtype=tf.float32)
    ix = tf.nn.top_k(scores, 6000, sorted=True,
                     name="top_anchors").indices
    window = np.array([0, 0, 1, 1], dtype=np.float32)

    anchor_index = []
    batch_idx = []
    box_offset = []
    batch_gt_class_ids = []


    for b in range(config.batch_size):
        org_idx = ix[b, :]

        gt_class_ids = input_gt_class_ids[b, :]
        gt_boxes = input_gt_boxes[b, :, :]

        scores_tp = tf.gather(scores[b, :], org_idx)
        deltas_tp = tf.gather(box_deltas[b, :], org_idx)
        pre_nms_anchors_tp = tf.gather(anchors, org_idx)

        boxes_tp = utils.apply_box_deltas_graph(pre_nms_anchors_tp, deltas_tp)
        boxes_tp = utils.clip_boxes_graph(pre_nms_anchors_tp, window)

        props = utils.nms(boxes_tp, scores_tp, cfg)

        proposals, _ = utils.trim_zeros_graph(props, name="trim_proposals")



        gt_boxes, non_zeros = utils.trim_zeros(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name="trim_gt_class_ids")


        overlaps = utils.overlaps_graph(proposals, gt_boxes)

        best_true = tf.argmax(overlaps, axis=0)
        roi_iou_max = tf.reduce_max(overlaps, axis=1)

        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]

        # positive_indices = tf.concat([best_true, positive_indices], axis=0)
        negative_indices = tf.where(roi_iou_max < 0.3)[:, 0]
        # TRAIN_ROIS_PER_IMAGE =200 ROI_POSITIVE_RATIO=0.33

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
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)+1

        # 将对应的ROI 与true_box 进行编码
        deltas = utils.box_encode(positive_rois, roi_gt_boxes)
        deltas /= cfg.BBOX_STD_DEV
        N = tf.shape(negative_rois)[0]
        deltas = tf.pad(deltas, [(0, N), (0, 0)])

        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N)])

        indeces = tf.concat([positive_indices, negative_indices],axis=0)

        org_idx = tf.gather(org_idx, indeces)
        bch_idx = tf.ones(shape=[tf.shape(indeces)[0]],dtype=tf.int32)*b


        batch_gt_class_ids.append(roi_gt_class_ids)
        batch_idx.append(bch_idx)
        box_offset.append(deltas)
        anchor_index.append(org_idx)

    batch_gt_class_ids = tf.concat(batch_gt_class_ids, axis=0)
    batch_idx = tf.concat(batch_idx, axis=0)
    box_offset = tf.concat(box_offset, axis=0)
    anchor_index = tf.concat(anchor_index, axis=0)
    return batch_gt_class_ids, batch_idx, box_offset, anchor_index



def arm_class_loss(rpn_class_logits, input_rpn_class):
    anchor_class = tf.cast(tf.greater(input_rpn_class, 0), tf.int32)
    total_pos = tf.reduce_sum(anchor_class)

    indices = tf.where(tf.not_equal(input_rpn_class, -1))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    anchor_class = tf.one_hot(anchor_class,2)
    ls = loss.focal_loss(rpn_class_logits,anchor_class)
    return ls/ tf.cast(total_pos,tf.float32)

def arm_box_loss(pred_box_offset, true_box_offset, input_rpn_class):

    conf_t = tf.reshape(input_rpn_class, shape=(-1,))
    true_box_offset = tf.reshape(true_box_offset, shape=(-1, 4))
    positive_roi_ix = tf.where(conf_t > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(conf_t, positive_roi_ix), tf.int64)

    pred_box_offset = tf.reshape(pred_box_offset, shape=(-1, 4))
    target_bbox = tf.gather(true_box_offset, positive_roi_ix)
    pred_bbox = tf.gather(pred_box_offset, positive_roi_ix)

    target_bbox = tf.cast(target_bbox, tf.float32)
    ls = loss.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)
    loss_l = tf.reduce_mean(ls)

    return loss_l



def odm_class_loss(pred_conf, gt_labels):

    anchor_class = tf.cast(tf.greater(gt_labels, 0), tf.int32)
    total_pos = tf.reduce_sum(anchor_class)


    indices = tf.where(tf.greater(gt_labels, -1))

    pred_conf = tf.gather_nd(pred_conf, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    anchor_class = tf.one_hot(anchor_class, 2)
    ls = loss.focal_loss(pred_conf, anchor_class)

    return ls / tf.cast(total_pos, tf.float32)





def odm_box_loss(pred_loc, true_boxes_offset, gt_label):
    conf_t = tf.reshape(gt_label, shape=(-1,))
    true_boxes_offset = tf.reshape(true_boxes_offset, shape=(-1, 4))

    positive_roi_ix = tf.where(conf_t > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(conf_t, positive_roi_ix), tf.int64)

    pred_loc = tf.reshape(pred_loc, shape=(-1, 4))

    true_boxes_offset = tf.gather(true_boxes_offset, positive_roi_ix)
    pred_bbox = tf.gather(pred_loc, positive_roi_ix)
    ls = loss.smooth_l1_loss(true_boxes_offset, pred_bbox)
    ls = tf.reduce_mean(ls)
    return ls



def model(img, input_gt_boxes_offset, input_gt_rpn_labels):
    arm_fps, odm_fps = resnet50.fpn_retin_det(img)
    rpn_class_logits, rpn_probs, rpn_bbox = get_rpns(arm_fps)


    pred_conf, pred_loc = get_odm(odm_fps)

    new_box_offset, gt_labels = refine_box(rpn_probs, rpn_bbox, input_gt_rpn_labels, input_gt_boxes_offset)



    arm_class_losses = arm_class_loss(rpn_class_logits, input_gt_rpn_labels)
    arm_box_losses = arm_box_loss(rpn_bbox, input_gt_boxes_offset, input_gt_rpn_labels)

    odm_class_losses = odm_class_loss(pred_conf, gt_labels)
    odm_box_losses = odm_box_loss(pred_loc, new_box_offset, gt_labels)



    tf.losses.add_loss(arm_class_losses)
    tf.losses.add_loss(arm_box_losses)
    tf.losses.add_loss(odm_class_losses)
    tf.losses.add_loss(odm_box_losses)

    total_loss = tf.losses.get_losses()

    tf.summary.scalar(name='arm_class_losses',tensor=arm_class_losses)
    tf.summary.scalar(name='arm_box_losses', tensor=arm_box_losses)
    tf.summary.scalar(name='odm_class_losses', tensor=odm_class_losses)
    tf.summary.scalar(name='odm_box_losses', tensor=odm_box_losses)
    train_tensors = tf.identity(total_loss, 'ss')

    return train_tensors


def debug():
    from utils import np_utils
    from data_set import data_gen
    tf.enable_eager_execution()
    gen_bdd = data_gen.get_batch(batch_size=config.batch_size, class_name='guoshu', image_size=config.image_size,
                                 max_detect=100)
    images, true_box, true_label = next(gen_bdd)

    rpn_box, rpn_label = np_utils.get_loc_conf_new(true_box, true_label,
                                                   batch_size=config.batch_size,
                                                   cfg=config.Config)

    arm_fps, odm_fps = inceptionv2.fpn_retidet(images)
    rpn_class_logits, rpn_probs, rpn_bbox = get_rpns(arm_fps)

    new_box_offset, gt_labels = refine_box(rpn_probs, rpn_bbox, rpn_label, rpn_box)


    pred_conf, pred_loc = get_odm(odm_fps)
    arm_class_losses = arm_class_loss(rpn_class_logits, rpn_label)
    arm_box_losses = arm_box_loss(rpn_bbox, rpn_box, rpn_label)

    odm_class_losses = odm_class_loss(pred_conf, gt_labels)
    odm_box_losses = odm_box_loss(pred_loc, new_box_offset,gt_labels)
    print(arm_class_losses, arm_box_losses,odm_class_losses,odm_box_losses  )



if __name__ == '__main__':
    debug()









