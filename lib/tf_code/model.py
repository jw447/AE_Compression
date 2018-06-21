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

import re

import tensorflow as tf
import scipy.io as sio
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 40,
                            """Number of images to process in a batch.""")
import data_input
# import data_input_gradient as data_input

# if hasattr(FLAGS, 'num_tags'):
#     if FLAGS.num_tags == 17:
#         import data_input
#     elif FLAGS.num_tags == 118:
#         import data_input_118tags as data_input
# else:
#     import data_input

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = data_input.IMAGE_SIZE
# NUM_CLASSES = data_input.NUM_CLASSES
NUM_CLASSES = FLAGS.num_tags
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.8  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

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
                                        batch_size=FLAGS.batch_size, data_type=data_type, num_tags=FLAGS.num_tags, tag_id=FLAGS.tag_id, tag_id0=FLAGS.tag_id0, tag_id1=FLAGS.tag_id1)


def inputs(eval_data, leaveout=None, batch_size = None):
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
                              batch_size=batch_size, leaveout=leaveout, data_type=data_type, num_tags=FLAGS.num_tags, tag_id = FLAGS.tag_id,tag_id0=FLAGS.tag_id0, tag_id1=FLAGS.tag_id1)


def inference(images, architecutre='vgg_16'):
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
  depth = images.get_shape()[3].value
  print(batch_size)
  if architecutre == 'vgg_16':
    # conv1
    with tf.variable_scope('conv1') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[7, 7, depth, 64],
                                          stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1)

      # visualize the learned filter
      if True: #FLAGS.use_mag:
        filter_layer1_width = 8
        filter_layer1_height = 8
        filter_w = 7
        filter_d = depth
        filter_layer1 = tf.transpose(
          tf.reshape(kernel, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])),
          [4, 3, 0, 1, 2])
        filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
        for height_idx in range(0, filter_layer1_height):
          filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx],
                                                 [filter_layer1_width, filter_w, filter_w, filter_d])
          tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
          for width_idx in range(0, filter_layer1_width):
              tmp[width_idx] = tf.reshape(tmp[width_idx], [1, filter_w, filter_w, filter_d])
          filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
        filter_layer1 = tf.concat(0, filter_layer1)
        filter_layer1 = tf.reshape(filter_layer1, [1, filter_layer1_height * filter_w, filter_layer1_width * filter_w, filter_d])
        tf.image_summary('filter1', filter_layer1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128],
                                          stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv2)

    # norm2
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')

    # conv3
    with tf.variable_scope('conv3') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128],
                                         stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv3 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv3)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4
    with tf.variable_scope('conv4') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256],
                                           stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv4 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv4)

    # conv5
    with tf.variable_scope('conv5') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                           stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv5 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    # conv6
    with tf.variable_scope('conv6') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512],
                                           stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv6 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv6)

    # conv7
    with tf.variable_scope('conv7') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                           stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv7 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv7)

    # # conv8
    # with tf.variable_scope('conv8') as scope:
    #   kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
    #                                        stddev=0.04, wd=0.0)
    #   conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
    #   biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    #   bias = tf.nn.bias_add(conv, biases)
    #   conv8 = tf.nn.relu(bias, name=scope.name)
    #   _activation_summary(conv8)

    # pool8
    pool8 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool8')

    # conv9
    with tf.variable_scope('conv9') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                           stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv9 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv9)

    # conv10
    with tf.variable_scope('conv10') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                           stddev=0.04, wd=0.0)
      conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv10 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv10)

    # pool10
    pool10 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool10')

    # local11
    with tf.variable_scope('local11') as scope:
     # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool10, [batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 1000],
                                          stddev=0.04, wd=0.0)
      biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.1))
      local11 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      local11 = tf.nn.dropout(local11, 0.5)
      _activation_summary(local11)

    # # local12
    # with tf.variable_scope('local12') as scope:
    #   weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
    #                                       stddev=0.04, wd=0.0)
    #   biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
    #   local12 = tf.nn.relu(tf.matmul(local11, weights) + biases, name=scope.name)
    #   local12 = tf.nn.dropout(local12, 0.5)
    #   _activation_summary(local12)

    # sigmoid,
    with tf.variable_scope('sigmoid_linear') as scope:
      weights = _variable_with_weight_decay('weights', [1000, NUM_CLASSES],
                                           stddev=1/192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                               tf.constant_initializer(0.0))
      sigmoid_linear = tf.add(tf.matmul(local11, weights), biases, name=scope.name)
      _activation_summary(sigmoid_linear)

  # alex net
  elif architecutre=='alexnet':
   #conv1
    with tf.variable_scope('conv1') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[11, 11, 1, 64],
                                           stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1)

   # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1')
  # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 192],
                                               stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv2)

  # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')

  # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='VALID', name='pool2')

  # conv3
    with tf.variable_scope('conv3') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 192, 384],
                                           stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv3 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv3)

  # pool3
  # pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
  #                        strides=[1, 2, 2, 1], padding='SAME', name='pool3')

  # conv4
    with tf.variable_scope('conv4') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 384, 256],
                                           stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv4 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv4)

  # conv5
    with tf.variable_scope('conv5') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                           stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv5 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv5)

  # pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='VALID', name='pool5')


  # local6
    with tf.variable_scope('local6') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool5, [batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 4096],
                                            stddev=0.04, wd=0.0)
      biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
      local6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      local6 = tf.nn.dropout(local6, 0.5)
      _activation_summary(local6)

  # local7
    with tf.variable_scope('local7') as scope:
      weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                            stddev=0.04, wd=0.0)
      biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
      local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
      local7 = tf.nn.dropout(local7, 0.5)
      _activation_summary(local7)

  # sigmoid,
    with tf.variable_scope('sigmoid_linear') as scope:
      weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES],
                                            stddev=1 / 192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
      sigmoid_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
      _activation_summary(sigmoid_linear)



  elif architecutre == '5layer':
    with tf.variable_scope('conv1') as scope:
      weights = _variable_with_weight_decay('weights', shape = [11, 11, depth, 96],
                                            stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(images, weights, [1,4,4,1], padding='SAME')
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1)

      # visualize the learned filter
      filter_layer1_width = 8
      filter_layer1_height = 12
      filter_w = 11
      filter_d = depth
      filter_layer1 = tf.transpose(tf.reshape(weights, ([filter_w,filter_w,filter_d, filter_layer1_width, filter_layer1_height])), [4, 3, 0, 1, 2])
      filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
      for height_idx in range(0, filter_layer1_height):
        filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx], [filter_layer1_width, filter_w, filter_w, filter_d])
        tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
        for width_idx in range(0, filter_layer1_width):
          tmp[width_idx] = tf.reshape(tmp[width_idx], [filter_w,filter_w, filter_d])
        filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
      filter_layer1 = tf.concat(0, filter_layer1)
      filter_layer1 = tf.reshape(filter_layer1, [1, filter_layer1_height*filter_w, filter_layer1_width*filter_w, filter_d])
      tf.image_summary('filter1', filter_layer1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,4,4,1], strides=[1,2,2,1], padding='VALID', name='pool1')

    with tf.variable_scope('conv2') as scope:
      weights = _variable_with_weight_decay('weights', shape=[5,5,96,192], stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv2)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    with tf.variable_scope('conv3') as scope:
      weights = _variable_with_weight_decay('weights', shape=[3,3,192,384], stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, biases)
      conv3 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv3)

    with tf.variable_scope('conv4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[3, 3, 384, 256], stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, biases)
      conv4 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv4)

    with tf.variable_scope('conv5') as scope:
      weights = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(conv4, weights, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, biases)
      conv5 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv5)

    pool3 = tf.nn.max_pool(conv5, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

    with tf.variable_scope('local4') as scope:
      reshape = tf.reshape(pool3, [batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 4000], stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [4000], tf.constant_initializer(0.1))
      local4 = tf.nn.relu( tf.matmul(reshape, weights) + biases, name=scope.name)
      _activation_summary(local4)

    with tf.variable_scope('local5') as scope:
        # reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
        # dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[4000, 4000], stddev=0.01, wd=0.0)
        biases = _variable_on_cpu('biases', [4000], tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
        _activation_summary(local5)

    with tf.variable_scope('sigmoid_linear') as scope:
      # reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
      # dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[4000, NUM_CLASSES], stddev=0.01, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
      sigmoid_linear = tf.matmul(local5, weights) + biases
      _activation_summary(sigmoid_linear)

  return sigmoid_linear

def inference_getfc(images, architecutre='vgg_16'):
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

    if architecutre == 'vgg_16':
        # conv1
        with tf.variable_scope('conv1') as scope:
            filters_in = int(images.get_shape()[-1])
            kernel = _variable_with_weight_decay('weights', shape=[7, 7, filters_in, 64],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)

            # visualize the learned filter
            filter_layer1_width = 8
            filter_layer1_height = 8
            filter_w = 7
            filter_d = filters_in
            filter_layer1 = tf.transpose(
                tf.reshape(kernel, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])),
                [4, 3, 0, 1, 2])
            filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
            for height_idx in range(0, filter_layer1_height):
                filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx],
                                                       [filter_layer1_width, filter_w, filter_w, filter_d])
                tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
                for width_idx in range(0, filter_layer1_width):
                    tmp[width_idx] = tf.reshape(tmp[width_idx], [1, filter_w, filter_w, filter_d])
                filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
            filter_layer1 = tf.concat(0, filter_layer1)
            filter_layer1 = tf.reshape(filter_layer1,
                                       [1, filter_layer1_height * filter_w, filter_layer1_width * filter_w, filter_d])
            tf.image_summary('filter1', filter_layer1)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv2)

        # norm2
        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
        #                   name='norm2')

        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv3)

        # pool3
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv4)

        # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv5)

        # pool5
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        # conv6
        with tf.variable_scope('conv6') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv6)

        # conv7
        with tf.variable_scope('conv7') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv7 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv7)

        # # conv8
        # with tf.variable_scope('conv8') as scope:
        #   kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
        #                                        stddev=0.04, wd=0.0)
        #   conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
        #   biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        #   bias = tf.nn.bias_add(conv, biases)
        #   conv8 = tf.nn.relu(bias, name=scope.name)
        #   _activation_summary(conv8)

        # pool8
        pool8 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool8')

        # conv9
        with tf.variable_scope('conv9') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv9 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv9)

        # conv10
        with tf.variable_scope('conv10') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                                 stddev=0.04, wd=0.0)
            conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv10 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv10)

        # pool10
        pool10 = tf.nn.max_pool(conv10, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME', name='pool10')

        # local11
        with tf.variable_scope('local11') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool10, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, 1000],
                                                  stddev=0.04, wd=0.0)
            biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.1))
            local11 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            local11 = tf.nn.dropout(local11, 0.5)
            _activation_summary(local11)

        # # local12
        # with tf.variable_scope('local12') as scope:
        #   weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
        #                                       stddev=0.04, wd=0.0)
        #   biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        #   local12 = tf.nn.relu(tf.matmul(local11, weights) + biases, name=scope.name)
        #   local12 = tf.nn.dropout(local12, 0.5)
        #   _activation_summary(local12)

        # sigmoid,
        with tf.variable_scope('sigmoid_linear') as scope:
            weights = _variable_with_weight_decay('weights', [1000, NUM_CLASSES],
                                                  stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            sigmoid_linear = tf.add(tf.matmul(local11, weights), biases, name=scope.name)
            # _activation_summary(sigmoid_linear)

        return local11, sigmoid_linear

    # alex net
    elif architecutre == 'alexnet':
        # conv1
        with tf.variable_scope('conv1') as scope:
            filters_in = int(images.get_shape()[-1])
            kernel = _variable_with_weight_decay('weights', shape=[11, 11, filters_in, 64],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)

            # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 192],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv2)

            # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')

        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 192, 384],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv3)

            # pool3
            # pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
            #                        strides=[1, 2, 2, 1], padding='SAME', name='pool3')

            # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 384, 256],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv4)

            # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                                 stddev=1e-4, wd=0.0)
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv5)

            # pool5
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        # local6
        with tf.variable_scope('local6') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool5, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, 4096],
                                                  stddev=0.04, wd=0.0)
            biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
            local6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            local6 = tf.nn.dropout(local6, 0.5)
            _activation_summary(local6)

            # local7
        with tf.variable_scope('local7') as scope:
            weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                                  stddev=0.04, wd=0.0)
            biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
            local7 = tf.nn.relu(tf.matmul(local6, weights) + biases, name=scope.name)
            local7 = tf.nn.dropout(local7, 0.5)
            _activation_summary(local7)

            # sigmoid,
        with tf.variable_scope('sigmoid_linear') as scope:
            weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES],
                                                  stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            sigmoid_linear = tf.add(tf.matmul(local7, weights), biases, name=scope.name)
            _activation_summary(sigmoid_linear)



    elif architecutre == '5layer':
        with tf.variable_scope('conv1') as scope:
            filters_in = int(images.get_shape()[-1])
            weights = _variable_with_weight_decay('weights', shape=[11, 11, filters_in, 96],
                                                  stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(images, weights, [1, 4, 4, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv1)

            # visualize the learned filter
            filter_layer1_width = 8
            filter_layer1_height = 12
            filter_w = 11
            filter_d = filters_in
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
                filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
            filter_layer1 = tf.concat(0, filter_layer1)
            filter_layer1 = tf.reshape(filter_layer1,
                                       [1, filter_layer1_height * filter_w, filter_layer1_width * filter_w, filter_d])
            tf.image_summary('filter1', filter_layer1)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        with tf.variable_scope('conv2') as scope:
            weights = _variable_with_weight_decay('weights', shape=[5, 5, 96, 192], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv2)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        with tf.variable_scope('conv3') as scope:
            weights = _variable_with_weight_decay('weights', shape=[3, 3, 192, 384], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv3)

        with tf.variable_scope('conv4') as scope:
            weights = _variable_with_weight_decay('weights', shape=[3, 3, 384, 256], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv4)

        with tf.variable_scope('conv5') as scope:
            weights = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(conv4, weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope.name)
            _activation_summary(conv5)

        pool3 = tf.nn.max_pool(conv5, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')

        with tf.variable_scope('local4') as scope:
            reshape = tf.reshape(pool3, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, 4000], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [4000], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            _activation_summary(local4)

        with tf.variable_scope('local5') as scope:
            # reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
            # dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[4000, 4000], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [4000], tf.constant_initializer(0.1))
            local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
            _activation_summary(local5)

        with tf.variable_scope('sigmoid_linear') as scope:
            # reshape = tf.reshape(pool3, [FLAGS.batch_size, -1])
            # dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[4000, NUM_CLASSES], stddev=0.01, wd=0.0)
            biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
            sigmoid_linear = tf.matmul(local5, weights) + biases
            _activation_summary(sigmoid_linear)

        return local5, sigmoid_linear

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


def loss_weight(logits, labels):
  """Add L2Loss to all the trainable variables.
  weight according to the number of positive and negative

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 2-D tensor
            of shape [batch_size, NUM_TAGS]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  batch_size = labels.get_shape()[0].value
  label_dis = sio.loadmat(os.path.join(FLAGS.data_dir, 'label_dis.mat'))
  label_dis = label_dis['train_labels']
  # the size is [num_tags]
  label_num_1s = np.sum(label_dis,0)
  label_num_0s = label_dis.shape[0] - label_num_1s

  pos_weight = np.tile(label_num_0s / label_num_1s, [batch_size,1])
  labels = tf.cast(labels, tf.float32)
  cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
      logits, labels, pos_weight, name='weighted_sigmoid_cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='weighted_sigmoid_cross_entropy')
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
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
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
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

