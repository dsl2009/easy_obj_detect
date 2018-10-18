import tensorflow as tf
import utils
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def soft_focal_loss(logits,labels,number_cls=20):
    labels = tf.one_hot(labels,number_cls)
    loss = tf.reduce_sum(labels*(-(1 - tf.nn.softmax(logits))**1*tf.log(tf.nn.softmax(logits))),axis=1)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):


    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)

    indices = tf.where(tf.not_equal(rpn_match, -1))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    loss = soft_focal_loss(labels=anchor_class,logits=rpn_class_logits ,number_cls=2)
    loss = tf.keras.backend.switch(tf.cast(tf.size(loss) > 0,tf.bool), tf.reduce_mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph( input_rpn_deltas, input_rpn_label, pred_rpn_deltas):

    input_rpn_deltas = tf.reshape(input_rpn_deltas,(-1,4))
    pred_rpn_deltas = tf.reshape(pred_rpn_deltas, (-1,4))
    input_rpn_label = tf.reshape(input_rpn_label, (-1,))

    indices = tf.where(tf.equal(input_rpn_label, 1))

    true_rpn_delatas = tf.gather(input_rpn_deltas, indices)
    pred_rpn_deltas = tf.gather(pred_rpn_deltas, indices)

    diff = tf.abs(true_rpn_delatas - pred_rpn_deltas)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.keras.backend.switch(tf.cast(tf.size(loss) > 0,tf.bool), tf.reduce_mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits):

    target_class_ids = tf.reshape(target_class_ids, shape=(-1,))

    target_class_ids = tf.cast(target_class_ids, 'int64')

    loss = soft_focal_loss(labels=target_class_ids, logits=pred_class_logits, number_cls=21)

    loss = tf.reduce_mean(loss)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):

    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)

    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    loss = tf.keras.backend.switch(tf.cast(tf.size(target_bbox) > 0,tf.bool),
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss

