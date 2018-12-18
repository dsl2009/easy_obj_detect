import tensorflow as tf
import utils
import config
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
    loss = tf.reduce_sum(labels*(-(1 - tf.nn.softmax(logits))**2*tf.log(tf.nn.softmax(logits))),axis=1)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):

    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    total_pos = tf.reduce_sum(anchor_class)

    indices = tf.where(tf.not_equal(rpn_match, -1))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    loss = soft_focal_loss(labels=anchor_class,logits=rpn_class_logits ,number_cls=2)
    loss = tf.keras.backend.switch(tf.cast(tf.size(loss) > 0,tf.bool), tf.reduce_sum(loss)/tf.cast(total_pos,tf.float32), tf.constant(0.0))
    return loss

def rpn_class_loss_multi_task(rpn_match, rpn_class_logits):
    total_loss = []
    for x in range(config.batch_size):
        b_rpn_match = rpn_match[x]
        b_rpn_logits = rpn_class_logits[x]
        anchor_class = tf.cast(tf.greater(b_rpn_match, 0), tf.int32)
        pos_num = tf.reduce_sum(anchor_class)
        indices_neg = tf.where(tf.equal(b_rpn_match, 0))
        indices_pos = tf.where(tf.greater(b_rpn_match, 0))
        indices_neg = tf.random_shuffle(indices_neg)[:3*pos_num]
        total_index = tf.concat([indices_neg, indices_pos],axis=0)

        b_rpn_class_logits = tf.gather(b_rpn_logits, total_index)
        anchor_class = tf.gather(anchor_class, total_index)

        loss = soft_focal_loss(labels=anchor_class,logits=b_rpn_class_logits ,number_cls=2)
        loss = tf.keras.backend.switch(tf.cast(tf.size(loss) > 0,tf.bool), tf.reduce_sum(loss), tf.constant(0.0))
        loss = loss/(tf.cast(pos_num, tf.float32)+1e-5)
        total_loss.append(loss)
    return tf.reduce_mean(total_loss)

def rpn_class_loss_graph1(rpn_match, rpn_class_logits):


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

    indices = tf.where(tf.greater(input_rpn_label, 0))

    true_rpn_delatas = tf.gather(input_rpn_deltas, indices)
    pred_rpn_deltas = tf.gather(pred_rpn_deltas, indices)

    diff = tf.abs(true_rpn_delatas - pred_rpn_deltas)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    loss = tf.keras.backend.switch(tf.cast(tf.size(loss) > 0,tf.bool), tf.reduce_mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, rois):
    print(target_class_ids, pred_class_logits, rois)
    #pred_bbox = tf.reshape(rois, shape=(-1, 4))
    #ix = tf.where(tf.reduce_sum(tf.abs(pred_bbox),axis=1)>0)[:,0]

    #target_class_ids = tf.gather(target_class_ids, ix)
    #pred_class_logits = tf.gather(pred_class_logits,ix)


    tf.summary.scalar('target_shape', tf.shape(target_class_ids)[0])
    target_class_ids = tf.reshape(target_class_ids, shape=(-1,))
    target_class_ids = tf.cast(target_class_ids, 'int64')
    pos_num = tf.reduce_sum(tf.cast(tf.greater(target_class_ids,0),tf.int32))
    tf.summary.scalar('pos_num', pos_num)
    loss = soft_focal_loss(labels=target_class_ids, logits=pred_class_logits, number_cls=2)

    loss = tf.keras.backend.switch(tf.cast(tf.size(loss) > 0, tf.bool), tf.reduce_sum(loss)/tf.cast(pos_num,tf.float32), tf.constant(0.0))

    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):

    print(pred_bbox)
    pred_bbox = tf.reshape(pred_bbox,(-1, 2, 4))
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    # pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))


    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    print(target_bbox,pred_bbox)
    loss = tf.keras.backend.switch(tf.cast(tf.size(target_bbox) > 0,tf.bool),
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss




def mrcnn_bbox_loss_graph_dsl(target_bbox, target_class_ids, pred_bbox):

    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))

    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)

    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather(pred_bbox, positive_roi_ix)

    loss = tf.keras.backend.switch(tf.cast(tf.size(target_bbox) > 0,tf.bool),
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss