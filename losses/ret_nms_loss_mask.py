#coding=utf-8
import tensorflow as tf
from utils import np_utils
import config
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


def sigmoid_cross_entropy_balanced(logits, labels, name='cross_entropy_loss'):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """

    y = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)



def get_box_loss(conf_t,loc_t,pred_loc, pred_confs,cfg):

    anc = np_utils.pt_from_nms(config.anchor_gen(config.image_size))
    anchors = tf.constant(anc,dtype=tf.float32)

    conf_t = tf.reshape(conf_t,shape=(-1,))
    loc_t = tf.reshape(loc_t,shape=(-1,4))

    positive_roi_ix = tf.where(conf_t > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(conf_t, positive_roi_ix), tf.int64)

    #indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    pred_loc = tf.reshape(pred_loc,shape=(-1,4))

    target_bbox = tf.gather(loc_t, positive_roi_ix)
    pred_bbox = tf.gather(pred_loc, positive_roi_ix)

    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                   smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                                   tf.constant(0.0))
    loss_l = tf.reduce_sum(loss)

    pred_conf = tf.reshape(pred_confs,shape=(-1, cfg.Config['num_classes']))

    zeros = tf.zeros(shape=tf.shape(conf_t), dtype=tf.float32)
    conf_t_tm = tf.where(tf.less(conf_t, 0), zeros, conf_t)
    conf_t_tm = tf.cast(conf_t_tm,tf.int32)
    conf_t_tm  = tf.reshape(conf_t_tm ,shape=(-1,))
    index = tf.stack([tf.range(tf.shape(conf_t_tm)[0]),conf_t_tm],axis=1)

    loss_c = log_sum(pred_conf) - tf.expand_dims(tf.gather_nd(pred_conf, index),-1)

    loss_c = tf.reshape(loss_c,shape=(cfg.batch_size,-1))
    conf_t = tf.reshape(conf_t, shape=(cfg.batch_size, -1))

    zeros = tf.zeros(shape=tf.shape(loss_c),dtype=tf.float32)
    loss_c = tf.where(tf.not_equal(conf_t,0),zeros,loss_c)

    pos_num = tf.reduce_sum(tf.cast(tf.greater(conf_t,0),dtype=tf.int32),axis=1)
    ne_num = pos_num*3

    los = []
    for s in range(cfg.batch_size):
        loss_tt = loss_c[s,:]
        ne_index = tf.image.non_max_suppression(
            boxes=anchors,
            scores=loss_tt,
            max_output_size=ne_num[s],
            iou_threshold=0.5
        )


        #value,ne_index = tf.nn.top_k(loss_tt,k=ne_num[s])

        pos_ix = tf.where(conf_t[s,:] > 0)[:,0]
        pos_ix = tf.cast(pos_ix,tf.int32)

        ix = tf.concat([pos_ix,ne_index],axis=0)

        label = tf.gather(conf_t[s,:],ix)
        label = tf.cast(label,tf.int32)
        lb,_,ct = tf.unique_with_counts(label)
        tf.summary.histogram('lbs', values=lb)
        tf.summary.histogram('ct', values=ct)

        logits = tf.gather(pred_confs[s,:],ix)


        #label = tf.one_hot(label, depth=cfg.Config['num_classes'])
        ls = tf.keras.backend.switch(tf.size(label) > 0,
                                     soft_focal_loss(labels=label, logits=logits,number_cls=cfg.Config['num_classes']),
                                     tf.constant(0.0))
        #ls = get_aver_loss(logists=logits,labels=label,number_cls=cfg.Config['num_classes'])
        ls = tf.reduce_sum(ls)
        los.append(ls)

    num = tf.reduce_sum(pos_num)
    num = tf.cast(num,dtype=tf.float32)

    final_loss_c = tf.keras.backend.switch(num > 0,
                                           tf.reduce_sum(los) / num,
                                 tf.constant(0.0))


    '''
    final_loss_c = tf.keras.backend.switch(num > 0,
                                        tf.reduce_mean(los),
                              tf.constant(0.0))
    '''

    final_loss_l = tf.keras.backend.switch(num > 0,
                                           loss_l / num,
                                           tf.constant(0.0))
    final_loss_c = final_loss_c



    return final_loss_l, final_loss_c



def mrcnn_mask_loss(target_masks, pred_masks, target_class):

    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    target_class = tf.cast(target_class, tf.int32)
    ix = tf.concat([tf.reshape(tf.range(tf.shape(pred_masks)[0]), [-1, 1]), tf.reshape(target_class, [-1, 1])], axis=1)

    pred_masks = tf.gather_nd(pred_masks, ix)
    ik = tf.where(tf.greater(tf.reduce_sum(target_masks,axis=[1,2]),0))
    pred_masks = tf.squeeze(tf.gather(pred_masks, ik),1)
    target_masks = tf.squeeze(tf.gather(target_masks, ik),1)

    target_masks = tf.cast(target_masks, tf.float32)
    tf.summary.image('target_mask', tf.expand_dims(target_masks, -1))
    tf.summary.image('pred_masks', tf.expand_dims(tf.sigmoid(pred_masks), -1))

    loss = tf.keras.backend.switch(tf.cast(tf.size(target_masks) > 0, tf.bool),
                                   tf.nn.sigmoid_cross_entropy_with_logits(labels=target_masks, logits=pred_masks),
                                   tf.constant(0.0))
    loss = tf.reduce_mean(loss)*100
    '''
    loss = sigmoid_cross_entropy_balanced(pred_masks,labels=target_masks)
    '''
    return loss


def get_loss(conf_t,loc_t,pred_loc, pred_confs,cfg, target_masks, pred_masks, target_class):
    final_loss_l, final_loss_c = get_box_loss(conf_t,loc_t,pred_loc, pred_confs,cfg)
    target_masks = tf.reshape(target_masks, shape=(-1, config.mask_pool_shape[0],config.mask_pool_shape[0]))
    mask_loss = mrcnn_mask_loss(target_masks, pred_masks, target_class)


    tf.losses.add_loss(final_loss_c)
    tf.losses.add_loss(final_loss_l)
    tf.losses.add_loss(mask_loss)

    total_loss = tf.losses.get_losses()
    tf.summary.scalar(name='class_loss', tensor=final_loss_c)
    tf.summary.scalar(name='loc_loss', tensor=final_loss_l)
    tf.summary.scalar(name='mask_loss', tensor=mask_loss)

    train_tensors = tf.identity(total_loss, 'ss')

    return train_tensors