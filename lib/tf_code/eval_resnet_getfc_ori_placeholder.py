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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
## python eval.py --eval_data ori_test --num_examples 2832 --architecture 5layer
## python eval.py --run_once True --architecture vgg_16 --eval_batch_size 1 --eval_data ori_test --num_examples 2832
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from datetime import datetime
import math
import time
import sklearn.metrics as smetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
import numpy as np
import tensorflow as tf
from scipy import misc


import model_resnet as model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'ori_test',
                           """Either 'test' or 'train_eval' or 'ori_test' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('architecture', 'resnet','network architecture, could be vgg_16 or alexnet')
tf.app.flags.DEFINE_integer('eval_batch_size', 1,
                            """Number of batch size to evaluate.""")

checkpoint_dir = FLAGS.checkpoint_dir + 'tf_log_' + FLAGS.architecture + '_train'
eval_dir = FLAGS.eval_dir + 'tf_placeholder_log_' + FLAGS.architecture + '_' + FLAGS.eval_data

# checkpoint_dir = FLAGS.checkpoint_dir + 'ft_train_lo-' + str(FLAGS.leaveout) + '_log_' + FLAGS.architecture
# eval_dir = FLAGS.eval_dir + 'ft_test_lo-' + str(FLAGS.leaveout) + '_log_' + FLAGS.architecture
NUM_CLASSES = model.NUM_CLASSES
NUM_EXAMPLES = [54, 618, 447, 91, 209, 134, 192, 26, 167, 54, 133, 244, 463]
IMAGE_NAME = '../src/PatchwiseMethod/ScatteringData.mat'
IMAGE_FILE = sio.loadmat(IMAGE_NAME)
IMAGE_FILE = IMAGE_FILE['images']

# directory is :
# ../data/ScatteringDataset/data/OPV/OPV-2010Aug18/
# ../data/ScatteringDataset/data/OPV/2011Jan25-Dan_OPV/
# ../data/ScatteringDataset/data/OPV/OPV-2010Feb22/
# ../data/ScatteringDataset/data/OPV/2011June04-Jon_surfaces/
# ../data/ScatteringDataset/data/2011Jan28-BrentCarey
# ../data/ScatteringDataset/data/BCP_nanoparticles/2011June04-BCP_000g/
# ../data/ScatteringDataset/data/BCP_nanoparticles/2011Aug09-BCP_nt_and_holes2/
# ../data/ScatteringDataset/data/CFN_SoftBio_group/2011June04-Cubes_GISAXS/
# ../data/ScatteringDataset/data/CFN_SoftBio_group/2011Apr30-Cubes_for_synthesis-Akron1sample/
# ../data/ScatteringDataset/data/various_for_people/2012Feb24-Germack_NYU/
# ../data/ScatteringDataset/data/grating/Tmode-2011_08Aug/
# ../data/ScatteringDataset/data/Theanne_Schiros/2012June27-Rubrene_careful-also_Zak/
# ../data/ScatteringDataset/data/Theanne_Schiros/2012Apr29-Rubrene_BN_rotation/



def eval_once(saver, prob_op, fc_op, image_op, image224_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    prob_op: prob op.
    gt_op: ground truth op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.3

  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      fc_all = []
      images_all = []
      images_256_all = []
      images_1k_all = []
      # for exp_idx in range(len(IMAGE_FILE)):
      for exp_idx in range(7,8):
        print(exp_idx)
        print(IMAGE_FILE[exp_idx][0])
        for img_idx in range(len(IMAGE_FILE[exp_idx][0])):
          img_name = str(IMAGE_FILE[exp_idx][0][img_idx, 1][0])[3:]
          image = misc.imread(img_name)
          image_1k = image
          # image = np.log(image) / np.log(1.0414)
          # image[np.isinf(image)] = 0
          # image = image.astype('int16')

          image_resize = misc.imresize(image, [256, 256])

          image_resize = image_resize.astype('float')
          # resize will change the max and min value in a picture to a value in 0-255, map it back
          image = image_resize * image.max() / image_resize.max()
          # take the log
          image = np.log(image) / np.log(1.0414)
          image[np.isinf(image)] = 0
          image = image.astype('int16')

          # whitening image
          image = image_resize.reshape(256,256,1)

          # image = tf.image.resize_images(image, 224, 224)
          # float_image = tf.image.per_image_whitening(image)
          # float_image = float_image.reshape(1, 224, 224, 1)
          images_ = sess.run([image224_op], feed_dict={image_op:image})
          # fc_fea = sess.run([fc_op], feed_dict={image_op: image})
          # fc_all.append(fc_fea)
          images_all.append(images_)
          images_256_all.append(image)
          images_1k_all.append(image_1k)


      fc_all = np.array(fc_all)

      fc_metrics = dict()
      fc_metrics['images'] = images_all
      fc_metrics['images_1k'] = images_1k_all
      fc_metrics['images_256'] = images_256_all
      # fc_metrics['feature'] = fc_all
      # sio.savemat(os.path.join(eval_dir, FLAGS.architecture + '_' + FLAGS.eval_data + '_fc.mat'), fc_metrics)
      sio.savemat(os.path.join(eval_dir, FLAGS.architecture + '_' + FLAGS.eval_data + '_images.mat'), fc_metrics)

      # summary = tf.Summary()
      # summary.ParseFromString(sess.run(summary_op))
      # summary.value.add(tag='Precision', simple_value=precision)
      # summary.value.add(tag='mAP', simple_value=meanAP)
      # summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data
    image = tf.placeholder('float32', [256, 256, 1])
    resize_image = tf.image.resize_images(image, 224, 224)
    whiten_image = tf.image.per_image_whitening(resize_image)
    images = tf.reshape(whiten_image, [1, 224, 224, 1])
    # images, labels = model.inputs(eval_data=eval_data, leaveout=FLAGS.leaveout, batch_size=FLAGS.eval_batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    fcs, logits = model.inference_getfc(images, is_training=False)

    # Calculate predictions.
    prob_op = tf.sigmoid(logits)
    # top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     model.MOVING_AVERAGE_DECAY)
    # variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()
    #
    # summary_writer = tf.train.SummaryWriter(eval_dir, g)

    while True:
      eval_once(saver, prob_op, fcs, image, images)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  # if tf.gfile.Exists(eval_dir):
  #   tf.gfile.DeleteRecursively(eval_dir)
  # tf.gfile.MakeDirs(eval_dir)
  if not tf.gfile.Exists(eval_dir):
    tf.gfile.MakeDirs(eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
