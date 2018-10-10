import tensorflow as tf
from tensorflow.contrib import slim
def cornet_arg():
    batch_norm_params = {
        'is_training':True,
        'decay': 0.9997,
        'epsilon':1e-5,
        'scale':True
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding = 'SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params) as sc:
            return sc

def residual(x, out_dim, stride = 1, scope =None):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d],biases_initializer=None):
            x1 = slim.conv2d(x, num_outputs=out_dim, kernel_size=3, stride=stride )
            x2 = slim.conv2d(x1, num_outputs=out_dim, kernel_size=3, activation_fn=None, rate=2)
            x3 = slim.conv2d(x, num_outputs=out_dim, kernel_size=1, stride=stride, activation_fn=None)
            return slim.nn.relu(x2+x3)

def model_list(x, n=5, n_dim=None, repeats = None):
    if not n_dim:
        n_dim = [256, 256, 384, 384, 384, 512]
    if not repeats:
        repeats = [2, 2, 2, 2, 2, 4]
    cur_num = repeats[0]
    next_num = repeats[1]
    cur_dim = n_dim[0]
    next_dim = n_dim[1]
    up1 = slim.repeat(x, cur_num, residual, out_dim=cur_dim, scope='up1'+str(n))
    x = slim.max_pool2d(up1, kernel_size=2, stride=2)
    x = slim.repeat(x, cur_num, residual, out_dim=next_dim, scope='low1'+str(n))
    if n>1:
        x = model_list(x, n-1, n_dim[1:], repeats[1:])
    else:
        x = slim.repeat(x, next_num, residual, out_dim=next_dim, scope='low2'+str(n))
    x = slim.repeat(x, cur_num-1, slim.conv2d, num_outputs=cur_dim, kernel_size=3, scope='low3'+str(n))
    x = tf.image.resize_bilinear(x, tf.shape(x)[1:3]*2)
    return x+up1

def kps_fun(image):
    with tf.variable_scope('cornet'):
        with slim.arg_scope(cornet_arg()):
            inter = slim.conv2d(image, num_outputs=128, kernel_size=7, stride=2)
            inter = slim.conv2d(inter, num_outputs=256, kernel_size=3, stride=2)
            kps = model_list(inter)
            return kps

def fpn(image):
    fea = kps_fun(image)
    c2_1 = slim.conv2d(fea, num_outputs=96, kernel_size=1, rate=1)
    c2_2 = slim.conv2d(fea, num_outputs=256, kernel_size=3, rate=4)
    c2_3 = slim.conv2d(fea, num_outputs=256, kernel_size=3, rate=6)
    c2_4 = slim.conv2d(fea, num_outputs=256, kernel_size=3, rate=8)
    c21 = tf.concat([c2_1, c2_2, c2_3, c2_4], axis=3)
    c21 = slim.conv2d(c21, num_outputs=256, kernel_size=3, stride=2, activation_fn=None)
    return c21

