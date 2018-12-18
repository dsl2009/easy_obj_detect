import numpy as np
import tensorflow as tf
import litht_head_config as cfg
import config
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
  """
  Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, 15, 15) window.
  """

  base_anchor = np.array([1, 1, base_size, base_size]) - 1
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                       for i in range(ratio_anchors.shape[0])])
  return anchors


def _whctrs(anchor):
  """
  Return width, height, x center, and y center for an anchor (window).
  """

  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  """
  Given a vector of widths (ws) and heights (hs) around a center
  (x_ctr, y_ctr), output a set of anchors (windows).
  """

  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                       y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1),
                       y_ctr + 0.5 * (hs - 1)))
  return anchors


def _ratio_enum(anchor, ratios):
  """
  Enumerate a set of anchors for each aspect ratio wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  """
  Enumerate a set of anchors for each scale wrt an anchor.
  """

  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

def generate_anchors_opr(
        height, width, feat_stride, anchor_scales=(8, 16, 32),
        anchor_ratios=(0.5, 1, 2), base_size=16):
    anchors = generate_anchors(
        ratios=np.array(anchor_ratios), scales=np.array(anchor_scales),
        base_size=base_size)
    shift_x = tf.range(width, dtype=np.float32) * feat_stride
    shift_y = tf.range(height, dtype=np.float32) * feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shifts = tf.stack(
        (tf.reshape(shift_x, (-1, 1)), tf.reshape(shift_y, (-1, 1)),
         tf.reshape(shift_x, (-1, 1)), tf.reshape(shift_y, (-1, 1))))
    shifts = tf.transpose(shifts, [1, 0, 2])
    final_anc = tf.constant(anchors.reshape((1, -1, 4)), dtype=np.float32) + \
          tf.transpose(tf.reshape(shifts, (1, -1, 4)), (1, 0, 2))
    return tf.reshape(final_anc, (-1, 4))
def bbox_transform_inv(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = tf.expand_dims(boxes[:, 0] + 0.5 * widths, -1)
    ctr_y = tf.expand_dims(boxes[:, 1] + 0.5 * heights, -1)

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    widths = tf.expand_dims(widths, -1)
    heights = tf.expand_dims(heights, -1)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights

    # x1
    # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_x1 = pred_ctr_x - 0.5 * pred_w
    # y1
    # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    # x2
    # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    # y2
    # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = _concat_new_axis(pred_x1, pred_y1, pred_x2, pred_y2, 2)
    pred_boxes = tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1))
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    x1 = tf.maximum(tf.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    y1 = tf.maximum(tf.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    x2 = tf.maximum(tf.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    y2 = tf.maximum(tf.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)

    pred_boxes = _concat_new_axis(x1, y1, x2, y2, 2)
    pred_boxes = tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1))
    return pred_boxes
def _concat_new_axis(t1, t2, t3, t4, axis):
    return tf.concat(
        [tf.expand_dims(t1, -1), tf.expand_dims(t2, -1),
         tf.expand_dims(t3, -1), tf.expand_dims(t4, -1)], axis=axis)


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = tf.where((ws >= min_size) & (hs >= min_size))[:, 0]
    return keep


def proposal_opr(
        rpn_cls_prob, rpn_bbox_pred,  _feat_stride, anchors,
        num_anchors):
    """ Proposal_layer with tensors
    """


    if True:
        pre_nms_topN = 12000
        post_nms_topN = 2000
        nms_thresh = 0.7
        batch = cfg.train_batch_per_gpu
    else:
        pre_nms_topN = 6000
        post_nms_topN = 1000
        nms_thresh = 0.7
        batch = cfg.test_batch_per_gpu

    if True:
        scores = tf.reshape(rpn_cls_prob, (batch, -1, 2))
        scores = scores[:, :, 1]
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (batch, -1, 4))

    if True:
        rpn_bbox_pred *= cfg.RPN_NORMALIZE_STDS
        rpn_bbox_pred += cfg.RPN_NORMALIZE_MEANS

    min_size = 5


    batch_scores = []
    batch_proposals = []
    for b_id in range(batch):
        cur_scores = scores[b_id]
        cur_rpn_bbox_pred = rpn_bbox_pred[b_id]

        cur_scores = tf.squeeze(tf.reshape(cur_scores, (-1, 1)), axis=1)
        cur_proposals = bbox_transform_inv(anchors, cur_rpn_bbox_pred)
        cur_proposals = clip_boxes(cur_proposals, config.image_size)

        if min_size > 0:
            assert 'Set MIN_SIZE will make mode slow with tf.where opr'
            keep = filter_boxes(cur_proposals, min_size )
            cur_proposals = tf.gather(cur_proposals, keep, axis=0)
            cur_scores = tf.gather(cur_scores, keep, axis=0)

        if pre_nms_topN > 0:
            cur_order = tf.nn.top_k(cur_scores, pre_nms_topN, sorted=True)[1]
            cur_proposals = tf.gather(cur_proposals, cur_order, axis=0)
            cur_scores = tf.gather(cur_scores, cur_order, axis=0)

        if True:
            tf_proposals = cur_proposals + np.array([0, 0, 1, 1])
            keep = tf.image.non_max_suppression(
                tf_proposals, cur_scores, post_nms_topN, nms_thresh)


        cur_proposals = tf.gather(cur_proposals, keep, axis=0)
        cur_scores = tf.gather(cur_scores, keep, axis=0)

        batch_inds = tf.ones((tf.shape(cur_proposals)[0], 1)) * b_id
        rois = tf.concat((batch_inds, cur_proposals), axis=1)
        batch_proposals.append(rois)
        batch_scores.append(cur_scores)

    final_proposals = tf.concat(batch_proposals, axis=0)
    final_scores = tf.concat(batch_scores, axis=0)
    return final_proposals, final_scores