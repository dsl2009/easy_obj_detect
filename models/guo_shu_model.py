import tensorflow as tf

from tensorflow.contrib import slim
import config
from nets import inception_resnet_v2
from base_model import resnet50,resnet101,coor_resnet
from dsl_libs import coor
def mul_channel_arg_scope(weight_decay=0.00004,
                        use_batch_norm=False,
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
            feature_map = slim.repeat(feature_map,4,coor.conv2d,num_outputs=256,kernel_size=3,stride=1,scope='classfy_repeat')
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
            feature_map = slim.repeat(feature_map, 4, coor.conv2d, num_outputs=256, kernel_size=3, stride=1,scope='regression_repeat')
            #feature_map = slim.repeat(feature_map, 4, mid_cov, scope='regression_repeat')
        out_puts = slim.conv2d(feature_map, 4 * num_anchors, kernel_size=3, stride=1,scope='regression',activation_fn=None)
        out_puts = tf.reshape(out_puts, shape=(config.batch_size,-1, 4))

    return out_puts





def get_box_logits(img,cfg):
    c1, c2, c3 = coor_resnet.fpn_add(img)

    logits = []
    boxes = []
    for ix, fp in enumerate([c1, c2, c3]):
        logits.append(classfy_model(fp,0, config.aspect_num[ix]))
        boxes.append(regression_model(fp,0, config.aspect_num[ix]))
    logits = tf.concat(logits, axis=1)
    boxes = tf.concat(boxes, axis=1)
    return boxes,logits


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
    ix = tf.where(tf.greater(pp,0.2))[:,0]
    score = tf.gather(pp,ix)
    box = tf.gather(box,ix)
    cls = tf.gather(cls, ix)
    box = tf.clip_by_value(box,clip_value_min=0.0,clip_value_max=1.0)
    b1 = tf.concat([box[:, 1:2], box[:, 0:1], box[:, 3:], box[:, 2:3]], axis=1)
    keep = tf.image.non_max_suppression(
        scores=score,
        boxes=b1,
        score_threshold=0.5,
        iou_threshold=0.2,
        max_output_size=40000
    )
    return tf.gather(box,keep,name='boxes'),tf.gather(score,keep,name='score'),tf.gather(cls,keep,name='pred')

