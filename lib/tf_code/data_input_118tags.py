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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import parse_binary_data_118tags as parse_binary_data
import parse_binary_oridata_17tags
# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = parse_binary_data.NUM_TAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 200*500*0.8
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200*500*0.2
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL_ori = 2832

def read_data(filename_queue, datatype='synthetic'):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.
    datatype: 'ori' or 'synthetic'
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class DataRecord(object):
    pass
  result = DataRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  if datatype == 'synthetic':
    label_bytes = parse_binary_data.NUM_TAGS   # int16 encoding
    result.height = parse_binary_data.IMAGESIZE # 256
    result.width = parse_binary_data.IMAGESIZE
    result.depth = 1
  elif datatype == 'ori':
    label_bytes = parse_binary_oridata_17tags.NUM_TAGS  # int 16
    result.height = parse_binary_oridata_17tags.IMAGESIZE  # 1024
    result.width = parse_binary_oridata_17tags.IMAGESIZE
    result.depth = 1
  image_bytes = result.height * result.width * result.depth
  print(image_bytes)
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = (label_bytes + image_bytes) * 2  # int16 encoding

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # record_bytes = tf.decode_raw(value, tf.float32)
  record_bytes = tf.decode_raw(value, tf.int16)

  # The first bytes represent the label, which we convert from uint8->int32.
  # the size of label is [NUM_TAGS]
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  # result.float32image = tf.transpose(depth_major, [1, 2, 0])
  result.float32image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle, num_preprocess_threads=16):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 1-D Tensor of [NUM_TAGS] of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    labels: Labels. 2D tensor of [batch_size, NUM_TAGS] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.

  # num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=80000)

  # Display the training images in the visualizer.
  tf.image_summary('images', images, max_images=20)

  return images, label_batch


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 2D tensor of [batch_size, NUM_TAGS] size.
  """
  filenames = [os.path.join(data_dir, 'train_batch_118tags_%d.bin' % i)
               for i in range(0, 8)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_data(filename_queue)
  reshaped_image = read_input.float32image

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # reshape image
  distorted_image = tf.image.resize_images(reshaped_image, height, width)

  # Randomly crop a [height, width] section of the image.
  # distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  # distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  # distorted_image = tf.image.random_brightness(distorted_image,
  #                                              max_delta=63)
  # distorted_image = tf.image.random_contrast(distorted_image,
  #                                            lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size, leaveout = None):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: string, 'train_eval', 'test', 'ori_test'.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size, NUM_TAGS] size.
  """
  num_preprocess_threads = 16
  if eval_data == 'train_eval':
    filenames = [os.path.join(data_dir, 'train_batch_118tags_%d.bin' % i)
                 for i in range(0, 8)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    num_preprocess_threads = 1
  elif eval_data == 'test':
    filenames = [os.path.join(data_dir, 'val_batch_118tags_%d.bin' % i)
                 for i in range(0, 2)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  elif eval_data == 'ori_test':
    # filenames = [os.path.join(data_dir, 'ori_batch.bin')]
    filenames = [os.path.join(data_dir, 'ori_resize_log_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL_ori
  elif eval_data == 'ft_train':
    filenames = [os.path.join(data_dir, 'ori_resize_log_lo-%d_train.bin' % leaveout)]
    num_examples_per_epoch = 2600
  elif eval_data == 'ft_test':
    filenames = [os.path.join(data_dir, 'ori_resize_log_lo-%d_test.bin' % leaveout)]
    num_examples_per_epoch = 26

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  if eval_data == 'ori_test':
    read_input = read_data(filename_queue, 'ori')
  else:
    read_input = read_data(filename_queue)
  reshaped_image = read_input.float32image

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # enhance contrast
  # it won't work since for per_image_whitening, it is the same.
  # if eval_data == 'ori_test':
  #   reshaped_image = tf.image.adjust_contrast(reshaped_image, 1)

  resized_image = tf.image.resize_images(reshaped_image, height, width)
  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
  #                                                        width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False, num_preprocess_threads=num_preprocess_threads)
