"""
patch wise auto encoder model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
import tensorflow.contrib.layers as layers

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
# tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118')
# tf.app.flags.DEFINE_string('data_dir', '../data/SyntheticScatteringData3BIN_log_32',
#                            """Path to the CIFAR-10 data directory: ../data/SyntheticScatteringData3BIN_log_32, ../data/SyntheticScatteringData_multi_cropsBIN_log_32 """)

import patch_auto_encoder_data as data_input

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = data_input.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

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
    with tf.device('/gpu:0'):
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
    """Construct distorted input for training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    return data_input.distorted_inputs(data_dir=data_dir,
                                       batch_size=FLAGS.batch_size)

# modularized for convolutional layer
def _conv_layer(input, kernel_size, name, padding='SAME', visualize = False, BN = True, is_training = True):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=kernel_size,
                                             stddev=0.04, wd=0.0)
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding=padding)
        biases = _variable_on_cpu('biases', kernel_size[-1], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if BN:
            bias = layers.batch_norm(bias, updates_collections=None, is_training=is_training)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
        if visualize:
            # visualize the learned filter
            filter_layer1_width = 8
            filter_layer1_height = int(kernel_size[3]/8)
            filter_w = kernel_size[0]
            filter_d = kernel_size[2]
            filter_layer1 = tf.transpose(
                tf.reshape(kernel, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])),
                [4, 3, 0, 1, 2])
            filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
            for height_idx in range(0, filter_layer1_height):
                filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx],
                                                       [filter_layer1_width, filter_w, filter_w, filter_d])
                tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
                for width_idx in range(0, filter_layer1_width):
                    tmp[width_idx] = tf.reshape(tmp[width_idx], [filter_w, filter_w, filter_d])
                filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
            filter_layer1 = tf.concat(0, filter_layer1)
            filter_layer1 = tf.reshape(filter_layer1,
                                       [1, filter_layer1_height * filter_w, filter_layer1_width * filter_w, filter_d])
            tf.image_summary(name+'_filter', filter_layer1)
    return conv1

# modularized for fully conncected layer
def _fc_layer(input, kernel_size, name, drop_out=0.5):
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=kernel_size,
                                              stddev=0.03, wd=0.0)
        biases = _variable_on_cpu('biases', kernel_size[1], tf.constant_initializer(0))
        local1 = tf.nn.relu(tf.matmul(input, weights) + biases, name=scope.name)
        # local1 = kema.batch_norm(local1,updates_collections=None)
        # local1 = tf.nn.dropout(local1, drop_out)
        _activation_summary(local1)
    return local1

# modularized for conv_transpose layer
# use conv2d_transpose
def _conv_trans_layer(input, filter_size, output_maps, name, strides, padding='SAME', activation = True, output_shape = None, BN = True, is_training=True):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=filter_size,
                                             stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', filter_size[-2], tf.constant_initializer(0.0))
        batch_size = input.get_shape()[0].value #tf.shape(input)[0]
        # height = tf.shape(input)[1] * strides
        # width = tf.shape(input)[2] * strides
        if output_shape is None:
            height = input.get_shape()[1].value * strides
            width = input.get_shape()[2].value * strides
            output_shape = [batch_size, height, width, output_maps]
        conv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1,strides, strides, 1], padding=padding)
        if activation:
            bias = tf.nn.bias_add(conv, biases)
            if BN:
                bias = layers.batch_norm(bias, is_training=is_training, updates_collections=None)
            conv1 = tf.nn.relu(bias, name=scope.name)
        else:
            conv1 = tf.nn.bias_add(conv, biases, name=scope.name)
        _activation_summary(conv1)
    return conv1

# modularized for con_transpose layer
# use upsampling + convolution
def _conv_trans_layer2(input, filter_size, name, strides, padding='SAME', activation = True):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=filter_size,
                                             stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', filter_size[-1], tf.constant_initializer(0.0))
        # height = tf.shape(input)[1] * strides
        # width = tf.shape(input)[2] * strides
        height = input.get_shape()[1].value * strides
        width = input.get_shape()[2].value * strides
        resized = tf.image.resize_images(input, height, width)
        conv = tf.nn.conv2d(resized, kernel, [1,1,1,1], padding=padding)
        if activation:
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        else:
            conv1 = tf.nn.bias_add(conv, biases, name=scope.name)
        _activation_summary(conv1)
    return conv1

def inference(images, is_training = True):
    """
    Auto encoder model
    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      representation
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    batch_size = images.get_shape()[0].value
    depth = images.get_shape()[3].value
    print(batch_size)
    print(images)
    # conv1
    conv1 = _conv_layer(images,[3,3,depth,96], 'encoder_conv1', 'SAME', True, is_training=is_training)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='encoder_pool1')
    # conv2
    conv2 = _conv_layer(pool1, [3, 3, 96, 128], 'encoder_conv2', 'SAME', False, is_training=is_training)

    # conv3
    conv3 = _conv_layer(conv2, [3, 3, 128, 128], 'encoder_conv3', 'SAME', False, is_training=is_training)
    # conv3 = _conv_layer(conv2, [3, 3, 128, 256], 'encoder_conv3', 'SAME', False, is_training=is_training)

    # pool2
    pool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool2')
    # conv4
    conv4 = _conv_layer(pool2, [3, 3, 128, 256], 'encoder_conv4', 'SAME', False, is_training=is_training)
    # conv4 = _conv_layer(pool2, [3, 3, 256, 256], 'encoder_conv4', 'SAME', False, is_training=is_training)
    # conv5
    conv5 = _conv_layer(conv4, [3, 3, 256, 256], 'encoder_conv5', 'SAME', False, is_training=is_training)
    # conv5 = _conv_layer(conv4, [3, 3, 256, 512], 'encoder_conv5', 'SAME', False, is_training=is_training)
    # pool3
    pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool3')
    # conv6
    conv6 = _conv_layer(pool3, [3, 3, 256, 512], 'encoder_conv6', 'SAME', False, is_training=is_training)
    # conv6 = _conv_layer(pool3, [3, 3, 512, 1024], 'encoder_conv6', 'SAME', False, is_training=is_training)
    # pool4
    pool4 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool4')
    # conv7
    conv7 = _conv_layer(pool4, [2, 2, 512, 1024], 'encoder_conv7', 'SAME', False, is_training=is_training)
    # conv7 = _conv_layer(pool4, [2, 2, 1024, 2048], 'encoder_conv7', 'SAME', False, is_training=is_training)
    # pool5
    pool5 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool5')

    # reshape pool5 to vector
    reshape = tf.reshape(pool5, [batch_size, -1])
    # apply softmax
    soft_max = tf.nn.softmax(reshape, name='softmax')
    _activation_summary(soft_max)

    rep_size = pool5.get_shape()[2].value
    rep_map = pool5.get_shape()[3].value
    dec_reshape = tf.reshape(soft_max, [batch_size, rep_size, rep_size, rep_map])

    rep_bf = pool5
    rep_af = soft_max

    dec_conv7 = _conv_trans_layer(dec_reshape, [2, 2, 512, 1024], 512, 'decoder_conv7', 2, padding='SAME', is_training=is_training)
    dec_conv6 = _conv_trans_layer(dec_conv7, [3, 3, 256, 512], 256, 'decoder_conv6', 2, padding='SAME', is_training=is_training)
    dec_conv5 = _conv_trans_layer(dec_conv6, [3, 3, 256, 256], 256, 'decoder_conv5', 2, padding='SAME', is_training=is_training)
    dec_conv4 = _conv_trans_layer(dec_conv5, [3, 3, 128, 256], 128, 'decoder_conv4', 1, padding='SAME', is_training=is_training)
    dec_conv3 = _conv_trans_layer(dec_conv4, [3, 3, 128, 128], 128, 'decoder_conv3', 2, padding='SAME', is_training=is_training)
    dec_conv2 = _conv_trans_layer(dec_conv3, [3, 3, 96, 128], 96, 'decoder_conv2', 1, padding='SAME', is_training=is_training)
    dec_conv1 = _conv_trans_layer(dec_conv2, [3, 3, depth, 96], depth, 'decoder_conv1', 2, padding='SAME', activation=False, is_training=is_training)

    # dec_conv7 = _conv_trans_layer(dec_reshape, [2, 2, 1024, 2048], 1024, 'decoder_conv7', 2, padding='SAME',
    #                               is_training=is_training)
    # dec_conv6 = _conv_trans_layer(dec_conv7, [3, 3, 512, 1024], 512, 'decoder_conv6', 2, padding='SAME',
    #                               is_training=is_training)
    # dec_conv5 = _conv_trans_layer(dec_conv6, [3, 3, 256, 512], 256, 'decoder_conv5', 2, padding='SAME',
    #                               is_training=is_training)
    # dec_conv4 = _conv_trans_layer(dec_conv5, [3, 3, 256, 256], 256, 'decoder_conv4', 1, padding='SAME',
    #                               is_training=is_training)
    # dec_conv3 = _conv_trans_layer(dec_conv4, [3, 3, 128, 256], 128, 'decoder_conv3', 2, padding='SAME',
    #                               is_training=is_training)
    # dec_conv2 = _conv_trans_layer(dec_conv3, [3, 3, 96, 128], 96, 'decoder_conv2', 1, padding='SAME',
    #                               is_training=is_training)
    # dec_conv1 = _conv_trans_layer(dec_conv2, [3, 3, depth, 96], depth, 'decoder_conv1', 2, padding='SAME',
    #                               activation=False, is_training=is_training)

    return rep_bf, rep_af, dec_conv1 #,  dec_conv9, dec_conv8, dec_conv7, dec_conv6
    # return  dec_conv1

def get_cluster(rep_af, is_training = True):
    """
       Auto encoder model
       Args:
         images: Images returned from distorted_inputs() or inputs().

       Returns:
         representation
       """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    batch_size = rep_af.get_shape()[0].value
    print(batch_size)
    depth = 1
    dec_reshape = tf.reshape(rep_af, [batch_size, 1, 1, 1024])

    dec_conv7 = _conv_trans_layer(dec_reshape, [2, 2, 512, 1024], 512, 'decoder_conv7', 2, padding='SAME',
                                  is_training=is_training)
    dec_conv6 = _conv_trans_layer(dec_conv7, [3, 3, 256, 512], 256, 'decoder_conv6', 2, padding='SAME',
                                  is_training=is_training)
    dec_conv5 = _conv_trans_layer(dec_conv6, [3, 3, 256, 256], 256, 'decoder_conv5', 2, padding='SAME',
                                  is_training=is_training)
    dec_conv4 = _conv_trans_layer(dec_conv5, [3, 3, 128, 256], 128, 'decoder_conv4', 1, padding='SAME',
                                  is_training=is_training)
    dec_conv3 = _conv_trans_layer(dec_conv4, [3, 3, 128, 128], 128, 'decoder_conv3', 2, padding='SAME',
                                  is_training=is_training)
    dec_conv2 = _conv_trans_layer(dec_conv3, [3, 3, 96, 128], 96, 'decoder_conv2', 1, padding='SAME',
                                  is_training=is_training)
    dec_conv1 = _conv_trans_layer(dec_conv2, [3, 3, depth, 96], depth, 'decoder_conv1', 2, padding='SAME',
                                  activation=False, is_training=is_training)

    return dec_conv1  # ,  dec_conv9, dec_conv8, dec_conv7, dec_conv6
    # return  dec_conv1


def loss(reconstruct, input):
    """ MSE Loss

    Args:
      reconstruct: reconstructed input
      input: original input

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.

    error = tf.reduce_mean(tf.square(input-reconstruct), name='mean_squared_error')
    tf.add_to_collection('losses', error)

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

