# auto encoder model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
# import tensorflow.contrib.layers as kema

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 10,
#                             """Number of frames to process in a batch.""")
# tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
# tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118')
# tf.app.flags.DEFINE_string('data_dir', '../data/SyntheticScatteringData3BIN_log_32',
#                            """Path to the CIFAR-10 data directory: ../data/SyntheticScatteringData3BIN_log_32, ../data/SyntheticScatteringData_multi_cropsBIN_log_32 """)

# -------------------------------------------------------------------------------
# Todo1
import gmx_input

# Global constants describing the CIFAR-10 data set.
# IMAGE_SIZE = gmx_input.IMAGE_SIZE
# NUM_CLASSES = data_input.NUM_CLASSES
# NUM_CLASSES = FLAGS.num_tags
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = gmx_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = gmx_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
# -------------------------------------------------------------------------------

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 1.0     # The decay to use for the moving average.
# NUM_EPOCHS_PER_DECAY = 1      # Epochs after which learning rate decays.
<<<<<<< HEAD
# LEARNING_RATE_DECAY_FACTOR = 1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.
=======
LEARNING_RATE_DECAY_FACTOR = 1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2

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
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
<<<<<<< HEAD
    with tf.device('/cpu:0'):
=======
    with tf.device('/gpu:0'):
>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2
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


<<<<<<< HEAD
=======

>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2
# Todo2
def inputs():
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: string, 'train', 'test', 'ori_test'.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
      labels: Labels. 1D tensor of [batch_size, NUM_TAGS] size.

    Raises:
      ValueError: If no data_dir
    """
    # TODO:
    # Consider add batch_size in the future

    # if not batch_size:
    #     batch_size = FLAGS.batch_size

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir

<<<<<<< HEAD
    filenames = [os.path.join(data_dir,'md_%d_seg.txt' % i) for i in range(0,300)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('File ' + f + 'Not found.')
    # print(len(filenames))

    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
=======
    filenames = [os.path.join(data_dir,'md_%d_seg.txt' % i) for i in range(0,100)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('File ' + f + 'Not found.')
    print(filenames)

    filename_queue = tf.train.string_input_producer(filenames)
>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2

    # shape = [3,300]
    return gmx_input.read_data(filename_queue)


# modularized for convolutional layer
def _conv_layer(input, kernel_size, name, padding='SAME', visualize = False):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=kernel_size,
                                             stddev=0.04, wd=0.0) # [3,3,1,100]
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding=padding)
        biases = _variable_on_cpu('biases', kernel_size[-1], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
        # if visualize:
        #     # visualize the learned filter
        #     # filter_layer1_width = 8
        #     filter_layer1_width = 10
        #     filter_layer1_height = int(kernel_size[3]/10) # 10
        #     filter_w = kernel_size[0] # 3
        #     filter_d = kernel_size[2] # 1
        #     print(tf.reshape(kernel, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])))
        #     # --------------------------------------------------------------------------------------------------------------------------------------
        #     filter_layer1 = tf.transpose(tf.reshape(kernel, ([filter_w, filter_w, filter_d, filter_layer1_width, filter_layer1_height])),
        #         [4, 3, 0, 1, 2])
        #     filter_layer1 = tf.split(0, filter_layer1_height, filter_layer1)
        #     for height_idx in range(0, filter_layer1_height):
        #         filter_layer1[height_idx] = tf.reshape(filter_layer1[height_idx],
        #                                                [filter_layer1_width, filter_w, filter_w, filter_d])
        #         tmp = tf.split(0, filter_layer1_width, filter_layer1[height_idx])
        #         for width_idx in range(0, filter_layer1_width):
        #             tmp[width_idx] = tf.reshape(tmp[width_idx], [filter_w, filter_w, filter_d])
        #         filter_layer1[height_idx] = tf.transpose(tf.concat(0, tmp))
        #     filter_layer1 = tf.concat(0, filter_layer1)
        #     filter_layer1 = tf.reshape(filter_layer1,
        #                                [1, filter_layer1_height * filter_w, filter_layer1_width * filter_w, filter_d])
        #     tf.image_summary(name+'_filter', filter_layer1)
    return conv1


# modularized for fully conncected layer
def _fc_layer(input, kernel_size, name, drop_out=0.5):
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=kernel_size,
                                              stddev=0.03, wd=0.0)
        biases = _variable_on_cpu('biases', kernel_size[1], tf.constant_initializer(0))
        local1 = tf.nn.tanh(tf.matmul(input, weights) + biases, name=scope.name)
        _activation_summary(local1)
    return local1


# modularized for conv_transpose layer
# use conv2d_transpose
def _conv_trans_layer(input, filter_size, output_maps, name, strides, padding='SAME', activation = True, output_shape = None):
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
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
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
<<<<<<< HEAD
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name) # sigmod
=======
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2
        else:
            conv1 = tf.nn.bias_add(conv, biases, name=scope.name)
        _activation_summary(conv1)
    return conv1


<<<<<<< HEAD
=======

>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2
def inference_fconn(images):
    '''
    Auto encoder with fully connected layers
    Args:
        images: Images returned from inputs().

    Returns:
        representation, reconstruction
    '''
    # print(images)
    images = tf.transpose(images)
    # print(images)
    # print('------------------')
    # encoder
<<<<<<< HEAD
    fc1 = _fc_layer(images, [1000, 700], 'encoder_fc_layer1')
    fc2 = _fc_layer(fc1, [700, 500], 'encoder_fc_layer2')
    fc3 = _fc_layer(fc2, [500, 300], 'encoder_fc_layer3')
    fc4 = _fc_layer(fc3, [300, 100], 'encoder_fc_layer4')
    # fc5 = _fc_layer(fc4, [100, 50], 'encoder_fc_layer5')
    # print(fc3)

    rep = fc4
    # decoder
    # fc6 = _fc_layer(rep, [50, 100], 'decoder_fc_layer1')
    fc7 = _fc_layer(rep, [100, 300], 'decoder_fc_layer2')
    fc8 = _fc_layer(fc7, [300, 500], 'decoder_fc_layer3')
    fc9 = _fc_layer(fc8, [500, 700], 'decoder_fc_layer4')
    fc10 = _fc_layer(fc9, [700, 1000], 'decoder_fc_layer5')
    # print(fc6)
    return rep, fc10
=======
    fc1 = _fc_layer(images, [1000, 500], 'encoder_fc_layer1')
    fc2 = _fc_layer(fc1, [500, 200], 'encoder_fc_layer2')
    # fc3 = _fc_layer(fc2, [100, 50], 'encoder_fc_layer3')
    # print(fc3)

    rep = fc2
    # decoder
    # fc4 = _fc_layer(rep, [50, 100], 'decoder_fc_layer1')
    fc3 = _fc_layer(rep, [200, 500], 'decoder_fc_layer2')
    fc4 = _fc_layer(fc3, [500, 1000], 'decoder_fc_layer3')
    # print(fc6)
    return rep, fc4
>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2


def inference_conv(images):
    """
    Auto encoder model
    Args:
      images: Images returned from or inputs().

    Returns:
      representation, reconstruction
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # batch_size = images.get_shape()[0].value
    # depth = images.get_shape()[3].value
    depth = 1
    # print(batch_size)
    # initial: shape=(1, 300, 3, 1)
    # conv1
    print(images)
    conv1 = _conv_layer(images,[1,30,depth,30], 'encoder_conv1', 'SAME', True)
    print(conv1) # (1, 300, 3, 30)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 10, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='encoder_pool1')
    print(pool1) # (1, 150, 2, 30)
    # conv2
    conv2 = _conv_layer(pool1, [1, 5, 30, 60], 'encoder_conv2', 'SAME', False)
    print(conv2) # (1, 150, 2, 60)
    # conv3
    conv3 = _conv_layer(conv2, [1, 30, 60, 10], 'encoder_conv3', 'SAME', False)
    print(conv3) # (1, 150, 2, 10)
    # pool2
    pool2 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool2')
    print(pool2) # (1, 75, 1, 10)
    # conv4
    conv4 = _conv_layer(pool2, [1, 25, 10, 3], 'encoder_conv4', 'SAME', False)
    print(conv4) # (1, 75, 1, 3)
    # conv5
    conv5 = _conv_layer(conv4, [3, 25, 3, 1], 'encoder_conv5', 'SAME', False)
    print(conv5) # (1, 75, 1, 1)
    # pool3
    pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1],
                           strides=[1, 15, 15, 1], padding='SAME', name='encoder_pool3')
    print(pool3) # (1, 5, 1, 1)

    # # conv6 - output:
    # conv6 = _conv_layer(pool3, [50, 6, 1, 1], 'encoder_conv6', 'SAME', False)
    # print(conv6)
    # # pool4 - output: 30
    # pool4 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool4')
    # print(pool4)
    # # conv7 - output:
    # conv7 = _conv_layer(pool4, [3, 3, 256, 256], 'encoder_conv7', 'SAME', False)
    # print(conv7)
    # # pool5 - output:
    # pool5 = tf.nn.max_pool(conv7, ksize=[1, 2, 2, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool5')
    # print(pool5)
    # # conv8 - output:
    # conv8 = _conv_layer(pool5, [3, 3, 256, 512], 'encoder_conv8', 'SAME', False)
    # print(conv8)
    # # pool6 - output:
    # pool6 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool6')
    # print(pool6)
    # # conv9 - output:
    # conv9 = _conv_layer(pool6, [3, 3, 512, 1024], 'encoder_conv9', 'SAME', False)
    # print(conv9)
    # # pool7 - output:
    # pool7 = tf.nn.max_pool(conv9, ksize=[1, 2, 2, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='encoder_pool7')
    # print(pool7)

    rep = pool3
    # print(pool5)
    # fc1
    # reshape = tf.reshape(pool5, [batch_size, -1])
    # dim = reshape.get_shape()[1].value
    # rep_size = pool5.get_shape()[2].value
    # rep_map = pool5.get_shape()[3].value
    #
    # fc1 = _fc_layer(reshape, [dim, 4096], 'encoder_fc1')
    #
    # # fc2
    # fc2 = _fc_layer(fc1, [4096, 4096], 'encoder_fc2')
    # ## end of encoder
    # # print(fc2)
    #
    # # ## start of decoder
    # # dec_fc2 = _fc_layer(fc2, [4096,4096], 'decoder_fc2')
    # # dec_fc1 = _fc_layer(fc1, [4096, dim], 'decoder_fc1')
    # #
    # # dec_reshape = tf.reshape(dec_fc1, [batch_size, rep_size, rep_size, rep_map])
    # # (1, 15, 1, 1)
    # dec_conv9 = _conv_trans_layer(rep, [1, 2, 20, 1], 20, 'decoder_conv9', 2, padding='SAME')
    # print(dec_conv9) # (1, 30, 2, 20)

    # dec_conv8 = _conv_trans_layer(dec_conv9, [3, 3, 256, 19], 256, 'decoder_conv8', 2, padding='SAME', output_shape=[1, 7, 7, 256])
    # print(dec_conv8) # (1, 7, 7, 256)

    # dec_conv7 = _conv_trans_layer(dec_conv8, [3,3,256,256], 256, 'decoder_conv7', 2, padding='SAME')
    # print(dec_conv7) # (1, 14, 14, 256)
    # # rep = pool5
    # # dec_conv9 = _conv_trans_layer(pool7, [3, 3, 512, 1024], 512, 'decoder_conv9', 2, padding='SAME')
    # # dec_conv8 = _conv_trans_layer(pool6, [3, 3, 256, 512], 256, 'decoder_conv8', 2, padding='SAME',
    # #                               output_shape=[batch_size, 7, 7, 256])
    # # dec_conv7 = _conv_trans_layer(pool5, [3, 3, 256, 256], 256, 'decoder_conv7', 2, padding='SAME')

    # dec_conv6 = _conv_trans_layer(dec_conv7, [3, 3, 256, 256], 256, 'decoder_conv6', 2, padding='SAME')
    # print(dec_conv6)

    # dec_conv5 = _conv_trans_layer(dec_conv6, [3, 3, 128, 256], 128, 'decoder_conv5', 2, padding='SAME')
    # print(dec_conv5)

    # dec_conv4 = _conv_trans_layer(dec_conv5, [3, 3, 128, 128], 128, 'decoder_conv4', 1, padding='SAME')
    # print(dec_conv4)
    # # dec_conv4 = _conv_layer(dec_conv5, [3, 3, 256, 128], 'decoder_conv4', padding='SAME')
    # (1, 15, 1, 1)

    dec_conv3 = _conv_trans_layer(rep, [1, 1, 30, 1], 30, 'decoder_conv3', 5, padding='SAME')
    print(dec_conv3) # (1, 25, 5, 30)

    dec_conv2 = _conv_trans_layer(dec_conv3, [1, 3, 3, 30], 3, 'decoder_conv2', 2, padding='SAME')
    print(dec_conv2) # (1, 50, 10, 3)

    dec_conv1 = _conv_trans_layer(dec_conv2, [1, 3, 100, 3], 100, 'decoder_conv1', 2, padding='SAME', activation=False)
    print(dec_conv1) # (1, 100, 20, 100)
    #
    # dec_conv7 = _conv_trans_layer2(pool5, [3, 3, 512, 256], 'decoder_conv7', 2, padding='SAME')
    # dec_conv6 = _conv_trans_layer2(dec_conv7, [3, 3, 256, 256], 'decoder_conv6', 2, padding='SAME')
    # dec_conv5 = _conv_trans_layer2(dec_conv6, [3, 3, 256, 256], 'decoder_conv5', 2, padding='SAME')
    # dec_conv4 = _conv_trans_layer2(dec_conv5, [3, 3, 256, 128], 'decoder_conv4', 1, padding='SAME')
    # dec_conv3 = _conv_trans_layer2(dec_conv4, [3, 3, 128, 128], 'decoder_conv3', 2, padding='SAME')
    # dec_conv2 = _conv_trans_layer2(dec_conv3, [3, 3, 128, 96], 'decoder_conv2', 1, padding='SAME')
    # dec_conv1 = _conv_trans_layer2(dec_conv2, [3, 3, 96, depth], 'decoder_conv1', 2, padding='SAME', activation=False)

    # dec_conv1 = _conv_trans_layer2(pool1, [3, 3, 96, depth], 'decoder_conv1', 2, padding='SAME', activation=False)

    return rep, dec_conv1 #,  dec_conv9, dec_conv8, dec_conv7, dec_conv6
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
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train Auto Encoder model.

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
    # num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
<<<<<<< HEAD
    # decay_steps = 1
    # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                 global_step,
    #                                 decay_steps,
    #                                 LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)
    # tf.summary.scalar('learning_rate', lr)
    lr = INITIAL_LEARNING_RATE
=======
    decay_steps = 1
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

>>>>>>> 7049bf9b412ff27d33d5cdcbeb23d82b5ed0f9b2
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
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
