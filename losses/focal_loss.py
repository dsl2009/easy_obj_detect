#coding=utf-8
import tensorflow as tf
from utils import np_utils
import config
from losses.loss import focal_loss
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss
def log_sum(x):
    mx = tf.reduce_max(x)
    data = tf.log(tf.reduce_sum(tf.exp(x - mx), axis=1)) + mx
    return tf.reshape(data, (-1, 1))
def soft_focal_loss(logits,labels,number_cls=20):
    labels = tf.one_hot(labels,number_cls)
    loss = tf.reduce_sum(labels*(-(1 - tf.nn.softmax(logits))**1*tf.log(tf.nn.softmax(logits))),axis=1)
    return loss
def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def get_loss(conf_t,loc_t,pred_loc, pred_confs,cfg):




    indices = tf.where(tf.not_equal(conf_t, -1))
    rpn_class_logits = tf.gather_nd(pred_confs, indices)
    anchor_class = tf.gather_nd(conf_t, indices)
    anchor_class = tf.cast(anchor_class,tf.int32)
    anchor_class = tf.one_hot(anchor_class, 2)
    #final_loss_c = soft_focal_loss(labels=anchor_class, logits=rpn_class_logits, number_cls=2)
    final_loss_c = focal_loss(rpn_class_logits, anchor_class)
    #final_loss_c =  tf.reduce_mean(final_loss_c)


    conf_t = tf.reshape(conf_t,shape=(-1,))
    loc_t = tf.reshape(loc_t,shape=(-1,4))

    positive_roi_ix = tf.where(conf_t > 0)[:, 0]
    num_pos = tf.reduce_sum(tf.cast(conf_t>0,tf.int32))

    positive_roi_class_ids = tf.cast(
        tf.gather(conf_t, positive_roi_ix), tf.int64)

    #indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    pred_loc = tf.reshape(pred_loc,shape=(-1,4))

    target_bbox = tf.gather(loc_t, positive_roi_ix)
    pred_bbox = tf.gather(pred_loc, positive_roi_ix)

    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                   smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                                   tf.constant(0.0))
    loss_l = tf.reduce_sum(loss)/tf.cast(num_pos,tf.float32)
    final_loss_c = final_loss_c/tf.cast(num_pos,tf.float32)




    tf.losses.add_loss(final_loss_c)
    tf.losses.add_loss(loss_l)

    total_loss = tf.losses.get_losses()

    tf.summary.scalar(name='class_loss',tensor=final_loss_c)
    tf.summary.scalar(name='loc_loss', tensor=loss_l)
    train_tensors = tf.identity(total_loss, 'ss')

    return train_tensors