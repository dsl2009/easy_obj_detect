import tensorflow as tf
from tensorflow.contrib import slim
from dsl_libs import coor
import config
batch_norm_params = {
        'is_training':True,
        'decay': 0.9997,
        'epsilon':1e-5,
        'scale':True
    }


def resnet_arg_scope_group_norm(weight_decay=0.0001,
                                activation_fn=tf.nn.relu,
                               ):
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.group_norm,
            ):

        with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
            return arg_sc

def second_arg_bn():
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding = 'SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params) as sc:
            return sc

def second_arg_gn():

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.group_norm,
            padding = 'SAME') as sc:
        return sc


def identity_block(input_tensor, in_depth,rate,scope):
    with tf.variable_scope(scope):
        x = coor.conv2d(input_tensor, num_outputs=in_depth, kernel_size=1, stride=1)
        x = coor.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1, rate=rate)
        x = coor.conv2d(x, num_outputs=in_depth * 4, kernel_size=1, stride=1,activation_fn=None)
        x += input_tensor
        return tf.nn.relu(x)

def conv_block(input_tensor, in_depth,  rate,stride=2):
    x = coor.conv2d(input_tensor, num_outputs=in_depth, kernel_size=1, stride=stride)
    x = coor.conv2d(x, num_outputs=in_depth, kernel_size=3, stride=1, rate=rate)
    x = coor.conv2d(x, num_outputs=in_depth * 4, kernel_size=1, stride=1, activation_fn=None)
    shortcut = coor.conv2d(input_tensor, num_outputs=in_depth* 4, kernel_size=1, stride=stride,activation_fn=None)
    x +=shortcut
    return tf.nn.relu(x)

def resnet(inputs):
    #inputs = tf.pad(inputs,[[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1 = coor.conv2d(inputs, 64, 7, stride=2, padding='VALID', scope='conv7')


    # Stage 2
    conv2 = conv_block(conv1, 64, rate=1, stride=2)
    conv2 = slim.repeat(conv2, 2, identity_block, scope='block1', in_depth=64, rate=1)
    # Stage 3
    conv3 = conv_block(conv2, 128, rate=1, stride=2)
    conv3 = slim.repeat(conv3, 3, identity_block, scope='block2', in_depth=128, rate=1)

    # Stage 4
    conv4 = conv_block(conv3, 256, rate=1, stride=2)
    conv4 = slim.repeat(conv4, 8, identity_block, scope='block3', in_depth=256, rate=1)
    # Stage 5
    conv5 = conv_block(conv4, 512, rate=1, stride=2)
    conv5 = slim.repeat(conv5, 2, identity_block, scope='block4', in_depth=512, rate=1)
    return conv2, conv3, conv4, conv5


def fpn(inputs):
    with slim.arg_scope(second_arg_bn()):
        c1, c2, c3, c4 = resnet(inputs)
    print(c1, c2, c3, c4)
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            c4_1 = coor.conv2d_transpose(c4, num_outputs=512, kernel_size=4, stride=2)
            print(c4_1)
            c3_1 = coor.conv2d_transpose(tf.concat([c4_1, c3], axis=3), num_outputs=512, kernel_size=4, stride=2)
            c2_1 = coor.conv2d_transpose(tf.concat([c3_1, c2], axis=3), num_outputs=512, kernel_size=4, stride=2)
            c1_1 = coor.conv2d_transpose(tf.concat([c2_1, c1], axis=3), num_outputs=256, kernel_size=4, stride=2)
            out_put = coor.conv2d_transpose(c1_1, num_outputs=128, kernel_size=4, stride=2)
            #out_put = coor.conv2d_transpose(out_put, num_outputs=64, kernel_size=4, stride=2)

    out_put = coor.conv2d_transpose(out_put, num_outputs=1, kernel_size=3, stride=1, activation_fn=None,
                                    normalizer_fn=None)
    out_put_mask = slim.nn.sigmoid(out_put)

    '''
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(0.0001),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding = 'SAME'):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            #c0 = slim.conv2d(tf.concat([out_put_mask, img],axis=3), num_outputs=32, kernel_size=3, stride=2)
            c0 = slim.conv2d(out_put_mask, num_outputs=32, kernel_size=3, stride=2)
            c0 = slim.conv2d(c0, num_outputs=64, kernel_size=3, stride=2)
            c1 = slim.conv2d(c0, num_outputs=128, kernel_size=3, stride=2)
            c2 = slim.conv2d(c1, num_outputs=128, kernel_size=3, stride=2)
            c3 = slim.conv2d(c2, num_outputs=128, kernel_size=3, stride=2)
            c4 = slim.conv2d(c3, num_outputs=128, kernel_size=3, stride=2)
    '''
    with slim.arg_scope(second_arg_bn()):
        p5 = coor.conv2d(c3, 256, 1, activation_fn=None)
        p5_upsample = tf.image.resize_bilinear(p5, tf.shape(c2)[1:3])
        p5 = slim.nn.relu(p5)
        p5 = coor.conv2d(p5, 256, 3, rate=2)
        p5 = coor.conv2d(p5, 256, 3, activation_fn=None)

        p4 = coor.conv2d(c2, 256, 1, activation_fn=None)
        p4 = p4 + p5_upsample
        p4_upsample = tf.image.resize_bilinear(p4, tf.shape(c1)[1:3])
        p4 = slim.nn.relu(p4)
        p4 = coor.conv2d(p4, 256, 3, rate=2)
        p4 = coor.conv2d(p4, 256, 3, activation_fn=None)

        p3 = coor.conv2d(c1, 256, 1, activation_fn=None)
        p3 = p3 + p4_upsample
        p3 = slim.nn.relu(p3)
        p3 = coor.conv2d(p3, 256, 3, rate=2)
        p3 = coor.conv2d(p3, 256, 3, activation_fn=None)

        p6 = coor.conv2d(c4, 1024, kernel_size=1)
        p6 = coor.conv2d(p6, 256, 3, rate=2)
        p6 = coor.conv2d(p6, 256, kernel_size=3, stride=1, activation_fn=None)

        p7 = slim.nn.relu(p6)
        p7 = coor.conv2d(p7, 256, kernel_size=3, stride=2, activation_fn=None)

    return p3, p4, p5, p6, out_put, out_put_mask



if __name__ == '__main__':
    xs = tf.placeholder(shape=(1,512,512,3),dtype=tf.float32)
    print(resnet(xs))