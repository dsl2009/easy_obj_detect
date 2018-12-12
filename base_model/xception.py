import collections
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils

from tensorflow.contrib import slim


_DEFAULT_MULTI_GRID = [1, 1, 1]


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.
  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """


def fixed_padding(inputs, kernel_size, rate=1):

  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


@slim.add_arg_scope
def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          **kwargs):

  def _separable_conv2d(padding):
    """Wrapper for separable conv2d."""
    return slim.separable_conv2d(inputs,
                                 num_outputs,
                                 kernel_size,
                                 depth_multiplier=depth_multiplier,
                                 stride=stride,
                                 rate=rate,
                                 padding=padding,
                                 scope=scope,
                                 **kwargs)
  def _split_separable_conv2d(padding):
    """Splits separable conv2d into depthwise and pointwise conv2d."""
    outputs = slim.separable_conv2d(inputs,
                                    None,
                                    kernel_size,
                                    depth_multiplier=depth_multiplier,
                                    stride=stride,
                                    rate=rate,
                                    padding=padding,
                                    scope=scope + '_depthwise',
                                    **kwargs)
    return slim.conv2d(outputs,
                       num_outputs,
                       1,
                       scope=scope + '_pointwise',
                       **kwargs)
  if stride == 1 or not use_explicit_padding:
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='SAME')
    else:
      outputs = _split_separable_conv2d(padding='SAME')
  else:
    inputs = fixed_padding(inputs, kernel_size, rate)
    if regularize_depthwise:
      outputs = _separable_conv2d(padding='VALID')
    else:
      outputs = _split_separable_conv2d(padding='VALID')
  return outputs


@slim.add_arg_scope
def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None):

  if len(depth_list) != 3:
    raise ValueError('Expect three elements in depth_list.')
  if unit_rate_list:
    if len(unit_rate_list) != 3:
      raise ValueError('Expect three elements in unit_rate_list.')

  with tf.variable_scope(scope, 'xception_module', [inputs]) as sc:
    residual = inputs

    def _separable_conv(features, depth, kernel_size, depth_multiplier,
                        regularize_depthwise, rate, stride, scope):
      if activation_fn_in_separable_conv:
        activation_fn = tf.nn.relu
      else:
        activation_fn = None
        features = tf.nn.relu(features)
      return separable_conv2d_same(features,
                                   depth,
                                   kernel_size,
                                   depth_multiplier=depth_multiplier,
                                   stride=stride,
                                   rate=rate,
                                   activation_fn=activation_fn,
                                   regularize_depthwise=regularize_depthwise,
                                   scope=scope)
    for i in range(3):
      residual = _separable_conv(residual,
                                 depth_list[i],
                                 kernel_size=3,
                                 depth_multiplier=1,
                                 regularize_depthwise=regularize_depthwise,
                                 rate=rate*unit_rate_list[i],
                                 stride=stride if i == 2 else 1,
                                 scope='separable_conv' + str(i+1))
    if skip_connection_type == 'conv':
      shortcut = slim.conv2d(inputs,
                             depth_list[-1],
                             [1, 1],
                             stride=stride,
                             activation_fn=None,
                             scope='shortcut')
      outputs = residual + shortcut
    elif skip_connection_type == 'sum':
      outputs = residual + inputs
    elif skip_connection_type == 'none':
      outputs = residual
    else:
      raise ValueError('Unsupported skip connection type.')

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            outputs)


@slim.add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):

  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)
          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('stride', 1)

      # Collect activations at the block's end before performing subsampling.
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None):

  with tf.variable_scope(
      scope, 'xception', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + 'end_points'
    with slim.arg_scope([slim.conv2d,
                         slim.separable_conv2d,
                         xception_module,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if output_stride is not None:
          if output_stride % 2 != 0:
            raise ValueError('The output_stride needs to be a multiple of 2.')
          output_stride /= 2
        # Root block function operated on inputs.
        net = resnet_utils.conv2d_same(net, 32, 3, stride=2,
                                       scope='entry_flow/conv1_1')
        net = resnet_utils.conv2d_same(net, 64, 3, stride=1,
                                       scope='entry_flow/conv1_2')

        # Extract features for entry_flow, middle_flow, and exit_flow.
        net = stack_blocks_dense(net, blocks, output_stride)

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection, clear_collection=True)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
                             scope='prelogits_dropout')
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):

  if unit_rate_list is None:
    unit_rate_list = _DEFAULT_MULTI_GRID
  return Block(scope, xception_module, [{
      'depth_list': depth_list,
      'skip_connection_type': skip_connection_type,
      'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
      'regularize_depthwise': regularize_depthwise,
      'stride': stride,
      'unit_rate_list': unit_rate_list,
  }] * num_units)


def xception_41(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_41'):
  """Xception-41 model."""
  blocks = [
      xception_block('entry_flow/block1',
                     depth_list=[128, 128, 128],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block2',
                     depth_list=[256, 256, 256],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('entry_flow/block3',
                     depth_list=[728, 728, 728],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('middle_flow/block1',
                     depth_list=[728, 728, 728],
                     skip_connection_type='sum',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=8,
                     stride=1),
      xception_block('exit_flow/block1',
                     depth_list=[728, 1024, 1024],
                     skip_connection_type='conv',
                     activation_fn_in_separable_conv=False,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=2),
      xception_block('exit_flow/block2',
                     depth_list=[1536, 1536, 2048],
                     skip_connection_type='none',
                     activation_fn_in_separable_conv=True,
                     regularize_depthwise=regularize_depthwise,
                     num_units=1,
                     stride=1,
                     unit_rate_list=multi_grid),
  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope)




def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       activation_fn=tf.nn.relu,
                       regularize_depthwise=False,
                       use_batch_norm=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
  }
  if regularize_depthwise:
    depthwise_regularizer = slim.l2_regularizer(weight_decay)
  else:
    depthwise_regularizer = None
  with slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d],
      weights_initializer=tf.truncated_normal_initializer(
          stddev=weights_initializer_stddev),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope(
          [slim.conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.separable_conv2d],
            weights_regularizer=depthwise_regularizer) as arg_sc:
          return arg_sc




x = tf.placeholder(shape=[1,512,512,3],dtype=tf.float32)

pd, ed= xception_41(x, num_classes=10)
print(pd)
for d in ed:
    print(d, ed[d])