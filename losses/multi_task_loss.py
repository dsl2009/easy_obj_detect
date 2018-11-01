#coding=utf-8
import tensorflow as tf
from losses import rcnn_losses
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


def get_loss(conf_t,loc_t,pred_loc, pred_confs, rpn_deltas, rpn_logits,cfg):
    rpn_loc_loss = rcnn_losses.rpn_bbox_loss_graph(rpn_deltas, conf_t, loc_t)
    rpn_cls_loss = rcnn_losses.rpn_class_loss_multi_task(conf_t, rpn_logits)
    final_loss_c, final_loss_l = second_loss(conf_t,loc_t,pred_loc, pred_confs,cfg)

    tf.losses.add_loss(rpn_loc_loss)
    tf.losses.add_loss(rpn_cls_loss)

    tf.losses.add_loss(final_loss_c)
    tf.losses.add_loss(final_loss_l)
    total_loss = tf.losses.get_losses()
    train_tensors = tf.identity(total_loss, 'ss')
    return train_tensors


def second_loss(conf_t,loc_t,pred_loc, pred_confs,cfg):

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


    conf_t_tm = tf.cast(conf_t,tf.int32)
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
        value,ne_index = tf.nn.top_k(loss_tt,k=ne_num[s])
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



    return final_loss_c, final_loss_l