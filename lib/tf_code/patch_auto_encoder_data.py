
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import parse_binary_data
import parse_binary_oridata_17tags

IMAGE_SIZE = 32

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2832*1000

def read_data(filename_queue):
    """
     Args:
       filename_queue: A queue of strings with the filenames to read from.
     Returns:
       An object representing a single example, with the following fields:
         height: number of rows in the result (32)
         width: number of columns in the result (32)
         depth: number of color channels in the result (1)
         key: a scalar string Tensor describing the filename & record number
           for this example.
         float32image: a [height, width, depth] float32 Tensor with the image data
     """

    class DataRecord(object):
        pass
    result = DataRecord()

    result.height = IMAGE_SIZE  # 32
    result.width = IMAGE_SIZE
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    print(image_bytes)
    # Every record consists of a fixed number of bytes for each.
    # record_bytes = image_bytes * 4  # float32 encoding

    record_bytes = image_bytes * 4  # int32 encoding

    # Read a record, getting filenames from the filename_queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of float32 that is record_bytes long.
    # record_bytes = tf.decode_raw(value, tf.float32)

    record_bytes = tf.decode_raw(value, tf.int32)

    # The remaining bytes represent the image, which we reshape
    # from [height * width] to [height, width, depth ].
    depth_major = tf.reshape(tf.slice(record_bytes, [0], [image_bytes]),
                             [ result.height, result.width, result.depth])
    # result.float32image = depth_major

    result.float32image = tf.cast(depth_major, tf.float32)
    return result


def _generate_image_and_label_batch(image, min_queue_examples,
                                    batch_size, shuffle, num_preprocess_threads=16):
    """Construct a queued batch of images

    Args:
      image: 3-D Tensor of [height, width, 1] of type.float32.
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 1] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.

    if shuffle:
        images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 30 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images = tf.train.batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=80000)

    # Display the training images in the visualizer.
    tf.image_summary('images', images, max_images=20)

    return images


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    """
    # filenames = [os.path.join(data_dir, 'ori_patch_sample_%d.bin' % i)
    #                  for i in range(0, 10)]

    filenames = [os.path.join(data_dir, '2_ori_patch_sample_int32_%d.bin' % i)
                 for i in range(1, 11)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)


    read_input = read_data(filename_queue)
    reshaped_image = read_input.float32image

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # reshape image
    # distorted_image = tf.image.resize_images(reshaped_image, height, width)

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
    float_image = tf.image.per_image_whitening(reshaped_image)

    # float_image = reshaped_image
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

