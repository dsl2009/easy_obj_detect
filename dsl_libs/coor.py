import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow import nn
from tensorflow.python.ops import init_ops

def coords(input_tensor,with_r = False):
    b, h, w, c = input_tensor.shape.as_list()

    batch_size_tensor = tf.shape(input_tensor)[0]
    xx_ones = tf.ones([batch_size_tensor, w], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)

    xx_range = tf.tile(tf.expand_dims(tf.range(w), 0), [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, 1)

    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    yy_ones = tf.ones([batch_size_tensor, h], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(h), 0), [batch_size_tensor, 1])
    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)
    xx_channel = tf.cast(xx_channel, 'float32') / (w - 1)
    yy_channel = tf.cast(yy_channel, 'float32') / (h - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
    if with_r:
        rr = tf.sqrt(tf.square(xx_channel - 0.5) + tf.square(yy_channel - 0.5))
        ret = tf.concat([ret, rr], axis=-1)
    return ret

def conv2d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
    inputs = coords(inputs)
    return slim.conv2d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=stride,
                  padding='SAME',
                  data_format=data_format,
                  rate=rate,
                  activation_fn=activation_fn,
                  normalizer_fn=normalizer_fn,
                  normalizer_params=normalizer_params,
                  weights_initializer=weights_initializer,
                  weights_regularizer=weights_regularizer,
                  biases_initializer=biases_initializer,
                  biases_regularizer=biases_regularizer,
                  reuse=reuse,
                  variables_collections=variables_collections,
                  outputs_collections=outputs_collections,
                  trainable=trainable,
                  scope=scope)

def conv2d_transpose(inputs,
                            num_outputs,
                            kernel_size,
                            stride=1,
                            padding='SAME',
                            data_format='NHWC',
                            activation_fn=nn.relu,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_initializer=initializers.xavier_initializer(),
                            weights_regularizer=None,
                            biases_initializer=init_ops.zeros_initializer(),
                            biases_regularizer=None,
                            reuse=None,
                            variables_collections=None,
                            outputs_collections=None,
                            trainable=True,
                            scope=None):
    inputs = coords(inputs)
    return slim.conv2d_transpose(inputs,
                            num_outputs,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            data_format=data_format,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_initializer=weights_initializer,
                            weights_regularizer=weights_regularizer,
                            biases_initializer=biases_initializer,
                            biases_regularizer=biases_regularizer,
                            reuse=reuse,
                            variables_collections=variables_collections,
                            outputs_collections=outputs_collections,
                            trainable=trainable,
                            scope=scope)



if __name__ == '__main__':
    tf.enable_eager_execution()
    a = tf.random_uniform(shape=(2, 8,8,3),dtype=tf.float32)


