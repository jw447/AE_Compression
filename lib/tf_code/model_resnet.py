# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from config import Config

import re

import tensorflow as tf

import data_input
# import data_input_gradient as data_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../data/SyntheticScatteringData3BIN_log_32',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = data_input.IMAGE_SIZE
# NUM_CLASSES = data_input.NUM_CLASSES
NUM_CLASSES = FLAGS.num_tags
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
activation = tf.nn.relu

# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
# LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
# INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

LEARNING_RATE = 0.001
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  # if wd is not None:
  #   weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
  #   tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  data_type = FLAGS.data_type
  return data_input.distorted_inputs(data_dir=data_dir,
                                     batch_size=FLAGS.batch_size, data_type=data_type, num_tags=FLAGS.num_tags,
                                     tag_id=FLAGS.tag_id, tag_id0=FLAGS.tag_id0, tag_id1=FLAGS.tag_id1)


def inputs(eval_data, leaveout=None, batch_size = None, min_que = None, num_crops = None):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: string, 'train', 'test', 'ori_test'.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size, NUM_TAGS] size.

  Raises:
    ValueError: If no data_dir
  """
  if not batch_size:
      batch_size = FLAGS.batch_size
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  data_type = FLAGS.data_type
  return data_input.inputs(eval_data=eval_data, data_dir=data_dir,
                           batch_size=batch_size, leaveout=leaveout, data_type=data_type, num_tags=FLAGS.num_tags,
                           tag_id=FLAGS.tag_id, tag_id0=FLAGS.tag_id0, tag_id1=FLAGS.tag_id1, min_que=min_que, num_crops=num_crops)

  # return data_input.inputs(eval_data=eval_data, data_dir=data_dir,
  #                             batch_size=batch_size, leaveout=leaveout, min_que=min_que, num_crops=num_crops)


def inference(images, is_training, num_classes = NUM_CLASSES,
  num_blocks = [3, 4, 6, 3],  # defaults to 50-layer network
  use_bias = False,  # defaults to using batch norm
  bottleneck = True):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # vgg 16
  # batch_size = tf.shape(images)[0]
  batch_size = images.get_shape()[0].value


  c = Config()
  c['bottleneck'] = bottleneck
  c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
  c['ksize'] = 3
  c['stride'] = 1
  c['use_bias'] = use_bias
  c['fc_units_out'] = num_classes
  c['num_blocks'] = num_blocks
  c['stack_stride'] = 2

  with tf.variable_scope('scale1'):
      c['conv_filters_out'] = 64
      c['ksize'] = 7
      c['stride'] = 2
      # x = conv(images, c, visualize=(True and FLAGS.use_mag ))
      x = conv(images, c, visualize=True)
      x = bn(x, c)
      x = activation(x)

  with tf.variable_scope('scale2'):
      x = _max_pool(x, ksize=3, stride=2)
      c['num_blocks'] = num_blocks[0]
      c['stack_stride'] = 1
      c['block_filters_internal'] = 64
      x = stack(x, c)

  with tf.variable_scope('scale3'):
      c['num_blocks'] = num_blocks[1]
      c['block_filters_internal'] = 128
      assert c['stack_stride'] == 2
      x = stack(x, c)

  with tf.variable_scope('scale4'):
      c['num_blocks'] = num_blocks[2]
      c['block_filters_internal'] = 256
      x = stack(x, c)

  with tf.variable_scope('scale5'):
      c['num_blocks'] = num_blocks[3]
      c['block_filters_internal'] = 512
      x = stack(x, c)

  # post-net
  x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
  if num_classes != None:
      with tf.variable_scope('fc'):
        x = fc(x, c)


  return x

def inference_getfc(images, is_training, num_classes = NUM_CLASSES,
  num_blocks = [3, 4, 6, 3],  # defaults to 50-layer network
  use_bias = False,  # defaults to using batch norm
  bottleneck = True):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # vgg 16
  # batch_size = tf.shape(images)[0]
  batch_size = images.get_shape()[0].value


  c = Config()
  c['bottleneck'] = bottleneck
  c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
  c['ksize'] = 3
  c['stride'] = 1
  c['use_bias'] = use_bias
  c['fc_units_out'] = num_classes
  c['num_blocks'] = num_blocks
  c['stack_stride'] = 2

  with tf.variable_scope('scale1'):
      c['conv_filters_out'] = 64
      c['ksize'] = 7
      c['stride'] = 2
      x = conv(images, c, visualize=True)
      x = bn(x, c)
      x = activation(x)

  with tf.variable_scope('scale2'):
      x = _max_pool(x, ksize=3, stride=2)
      c['num_blocks'] = num_blocks[0]
      c['stack_stride'] = 1
      c['block_filters_internal'] = 64
      x = stack(x, c)

  with tf.variable_scope('scale3'):
      c['num_blocks'] = num_blocks[1]
      c['block_filters_internal'] = 128
      assert c['stack_stride'] == 2
      x = stack(x, c)

  with tf.variable_scope('scale4'):
      c['num_blocks'] = num_blocks[2]
      c['block_filters_internal'] = 256
      x = stack(x, c)

  with tf.variable_scope('scale5'):
      c['num_blocks'] = num_blocks[3]
      c['block_filters_internal'] = 512
      x = stack(x, c)

  # post-net
  x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
  x_fc = x
  if num_classes != None:
      with tf.variable_scope('fc'):
        x = fc(x, c)


  return  x_fc, x

def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias


    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x

def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)

def conv(x, c, visualize = False):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = int(x.get_shape()[-1])
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    if visualize:
        print('visualize filter')
        # visualize the learned filter
        filter_layer1_width = 8
        filter_layer1_height = 8
        filter_w = ksize
        filter_d = filters_in
        # print(shape)
        # print(tf.reshape(weights, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])))

        filter_layer1 = tf.transpose(
            tf.reshape(weights, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])),
            [4, 3, 0, 1, 2])
        filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
        for height_idx in range(0, filter_layer1_height):
            filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx],
                                                   [filter_layer1_width, filter_w, filter_w, filter_d])
            tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
            for width_idx in range(0, filter_layer1_width):
                tmp[width_idx] = tf.reshape(tmp[width_idx], [1, filter_w, filter_w, filter_d])
            # print(tf.transpose(tf.concat(0, tmp)))
            filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp), [1,2,0,3])
        filter_layer1 = tf.concat(0, filter_layer1)
        filter_layer1 = tf.reshape(filter_layer1,
                                   [1, filter_layer1_height * filter_w, filter_layer1_width * filter_w, filter_d])
        tf.image_summary('filter1', filter_layer1)

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 2-D tensor
            of shape [batch_size, NUM_TAGS]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.float32)
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      logits, labels, name='sigmoid_cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='sigmoid_cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  # variable_averages = tf.train.ExponentialMovingAverage(
  #     MOVING_AVERAGE_DECAY, global_step)
  # variables_averages_op = variable_averages.apply(tf.trainable_variables())
  #
  # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
  #   train_op = tf.no_op(name='train')

  batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
  batchnorm_updates_op = tf.group(*batchnorm_updates)
  train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

  return train_op

