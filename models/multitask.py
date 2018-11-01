import tensorflow as tf

from tensorflow.contrib import slim
import config
from nets import inception_resnet_v2
from base_model import resnet50,resnet101
def mul_channel_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu):

  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d]):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

def mul_channel_arg_scope_gropnorm(weight_decay=0.00004,
                        activation_fn=tf.nn.relu):
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d]):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=slim.group_norm,
        ) as sc:
      return sc

if config.is_use_group_norm:
    base_arg = mul_channel_arg_scope_gropnorm
else:
    base_arg = mul_channel_arg_scope

def mid_cov(x, scope):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        x1 = slim.conv2d(x, 128, 1)
        x1 = slim.conv2d(x1, 64, kernel_size=[5,1],stride=1)
        x1 = slim.conv2d(x1, 64, kernel_size=[1, 5], stride=1)
        f = slim.conv2d(x, 128, 1)
        f = tf.concat([x1,f],axis=3)
        f = slim.conv2d(f, 256, 3,activation_fn=None, normalizer_fn=None)
        x = x+f
        x = tf.nn.relu(x)
        return x


def classfy_model(feature_map,ix, num_anchors=9):
    with tf.variable_scope('classfy'+str(ix),reuse=tf.AUTO_REUSE):
        with slim.arg_scope(base_arg()):
            feature_map = slim.repeat(feature_map,4,slim.conv2d,num_outputs=256,kernel_size=3,stride=1,scope='classfy_repeat')
            #feature_map = slim.repeat(feature_map, 4, mid_cov, scope='classfy_repeat')
            #feature_map = mid_cov(feature_map, 'mid_'+str(ix))
        out_puts = slim.conv2d(feature_map, config.Config['num_classes'] * num_anchors, kernel_size=3, stride=1,scope='classfy_conv',
                               weights_initializer=tf.initializers.zeros,activation_fn=None)
        out_puts = tf.reshape(out_puts,shape=(config.batch_size,-1, config.Config['num_classes']))
        #out_puts = slim.nn.sigmoid(out_puts)
    return out_puts

def regression_model(feature_map,ix, num_anchors=9):
    with tf.variable_scope('regression'+str(ix), reuse=tf.AUTO_REUSE):
        with slim.arg_scope(base_arg()):
            feature_map = slim.repeat(feature_map, 4, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
            #feature_map = slim.repeat(feature_map, 4, mid_cov, scope='regression_repeat')
        out_puts = slim.conv2d(feature_map, 4 * num_anchors, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        out_puts = tf.reshape(out_puts, shape=(config.batch_size,-1, 4))

    return out_puts

def hebing(feature_map,scope,num_anchors=9):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        with slim.arg_scope(base_arg()):
            feature_map = slim.repeat(feature_map, 4, slim.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
        box = slim.conv2d(feature_map, 4 * num_anchors, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        box = tf.reshape(box, shape=(config.batch_size,-1, 4))

        logits = slim.conv2d(feature_map, config.Config['num_classes'] * num_anchors, kernel_size=3, stride=1,
                               scope='classfy_conv',
                               weights_initializer=tf.initializers.zeros, activation_fn=None)
        logits = tf.reshape(logits, shape=(config.batch_size, -1, config.Config['num_classes']))


    return box, logits

def rpn_graph(feature_map, anchors_per_location=config.aspect_num[0]):
    shared = slim.conv2d(feature_map, 512, 3, activation_fn=slim.nn.relu)
    x = slim.conv2d(shared, 2 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_class_logits = tf.reshape(x, shape=[tf.shape(x)[0], -1, 2])
    rpn_probs = slim.nn.softmax(rpn_class_logits)
    x = slim.conv2d(shared, 4 * anchors_per_location, kernel_size=1, padding='VALID', activation_fn=None)
    rpn_bbox = tf.reshape(x, shape=[tf.shape(x)[0], -1, 4])
    return [rpn_class_logits, rpn_probs, rpn_bbox]



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
    rpn_bbox = tf.concat(r_b, axis=1)
    return rpn_class_logits,  rpn_bbox




def get_box_logits(img,cfg):
    cn, bn = resnet50.fpn_retin_det(img)
    logits = []
    boxes = []
    for ix, fp in enumerate(bn):
        logits.append(classfy_model(fp,0, config.aspect_num[ix]))
        boxes.append(regression_model(fp,0, config.aspect_num[ix]))
    logits = tf.concat(logits, axis=1)
    boxes = tf.concat(boxes, axis=1)

    rpn_class_logits, rpn_bbox = get_rpns(bn)

    return boxes,logits, rpn_class_logits, rpn_bbox



def decode_box(prios,pred_loc,variance=None):
    if variance is None:
        variance =[0.1, 0.2]
    boxes = tf.concat((
        prios[:, :2] + pred_loc[:, :2] * variance[0] * prios[:, 2:],
        prios[:, 2:] * tf.exp(pred_loc[:, 2:] * variance[1])), 1)
    xy_min = boxes[:, :2] -boxes[:, 2:] / 2
    xy_max = boxes[:, 2:] + xy_min
    return  tf.concat([xy_min,xy_max],axis=1)

def predict(ig,pred_loc, pred_confs, cfg):
    priors = config.anchor_gen(config.image_size)
    box = decode_box(prios=priors, pred_loc=pred_loc[0])
    props = slim.nn.softmax(pred_confs[0])
    pp = props[:,1:]
    cls = tf.argmax(pp, axis=1)
    pp = tf.reduce_max(pp,axis=1)
    ix = tf.where(tf.greater(pp,0.3))[:,0]
    score = tf.gather(pp,ix)
    box = tf.gather(box,ix)
    cls = tf.gather(cls, ix)
    box = tf.clip_by_value(box,clip_value_min=0.0,clip_value_max=1.0)
    b1 = tf.concat([box[:, 1:2], box[:, 0:1], box[:, 3:], box[:, 2:3]], axis=1)
    keep = tf.image.non_max_suppression(
        scores=score,
        boxes=b1,
        iou_threshold=0.4,
        max_output_size=100
    )
    return tf.gather(box,keep),tf.gather(score,keep),tf.gather(cls,keep)

