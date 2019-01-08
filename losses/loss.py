import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)

    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)

    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)

def focal_loss1(logits, onehot_labels, alpha=0.25, gamma=2.0):
    """
    Compute sigmoid focal loss between logits and onehot labels: focal loss = -(1-pt)^gamma*log(pt)
    Args:
        onehot_labels: onehot labels with shape (batch_size, num_anchors, num_classes)
        logits: last layer feature output with shape (batch_size, num_anchors, num_classes)
        weights: weight tensor returned from target assigner with shape [batch_size, num_anchors]
        alpha: The hyperparameter for adjusting biased samples, default is 0.25
        gamma: The hyperparameter for penalizing the easy labeled samples, default is 2.0
    Returns:
        a scalar of focal loss of total classification
    """
    with tf.name_scope("focal_loss"):
        logits = tf.cast(logits, tf.float32)
        onehot_labels = tf.cast(onehot_labels, tf.float32)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)


        predictions = tf.sigmoid(logits)
        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
        # add small value to avoid 0
        alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        weighted_loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t
        return tf.reduce_sum(weighted_loss)
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
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
def soft_focal_loss(logits,labels,number_cls=20):
    labels = tf.one_hot(labels,number_cls)
    loss = tf.reduce_sum(labels*(-(1 - tf.nn.softmax(logits))**2*tf.log(tf.nn.softmax(logits))),axis=1)
    return loss

if __name__ == '__main__':
    tf.enable_eager_execution()
    t = tf.constant([[1,8.0],[1,8.0],[1,8.0],[1,8.0],[1,8.0]],tf.float32)
    print(tf.nn.softmax(t))
    b = [[0,1],[0,1],[0,1],[0,1],[1,1]]
    print(t>0)
    print(tf.reduce_sum(tf.cast(t>1,tf.int32)))
    print(tf.nn.softmax_cross_entropy_with_logits_v2(labels=b, logits=t))
    print(focal_loss(t,b))
    print(focal_loss1(t, b))