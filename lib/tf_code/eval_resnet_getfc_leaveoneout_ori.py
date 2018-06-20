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

import model_resnet as model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'ft_test',
                           """Either 'test' or 'train_eval' or 'ori_test' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('architecture', 'resnet','network architecture, could be vgg_16 or alexnet')
tf.app.flags.DEFINE_integer('leaveout', 0,
                           """number 0-12 .""")
tf.app.flags.DEFINE_integer('eval_batch_size', 1,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_integer('num_crops', 39,
                           """number 0-12 .""")
checkpoint_dir = FLAGS.checkpoint_dir + 'tf_log_' + FLAGS.architecture + '_train'
eval_dir = FLAGS.eval_dir + 'tf_leaveout_log_3' + FLAGS.architecture + '_' + FLAGS.eval_data

# checkpoint_dir = FLAGS.checkpoint_dir + 'ft_train_lo-' + str(FLAGS.leaveout) + '_log_' + FLAGS.architecture
# eval_dir = FLAGS.eval_dir + 'ft_test_lo-' + str(FLAGS.leaveout) + '_log_' + FLAGS.architecture
print(FLAGS.leaveout)
NUM_CLASSES = model.NUM_CLASSES
# num_crops = 11
NUM_EXAMPLES = np.array([54, 618, 447, 91, 209, 134, 192, 26, 167, 54, 133, 244, 463]) * FLAGS.num_crops

# for TSAXS data
if FLAGS.eval_data == 'TSAXSTWAXS_ft_test':
  NUM_EXAMPLES = [209,     1,     2,   114,     2,     3,     6,     6,    54,     3,    4,    4,     8]
elif FLAGS.eval_data == 'TSAXS_ft_test':
  NUM_EXAMPLES = [94,     1,     2,   114,     2,     2,     2,     2,    35,     2,     2,     6]

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



def eval_once(saver, prob_op, gt_op, fc_op):
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
      print(FLAGS.leaveout)
      num_iter = int(math.ceil(NUM_EXAMPLES[FLAGS.leaveout] / FLAGS.eval_batch_size))
      num_examples = num_iter * FLAGS.eval_batch_size
      print(num_examples)
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_examples * NUM_CLASSES
      pred_prob_all = np.zeros([num_examples, NUM_CLASSES])
      gt_all = np.zeros([num_examples, NUM_CLASSES])

      fc_dim = fc_op.get_shape()[1].value

      fc_all = np.zeros([num_examples, fc_dim])
      loss = 0
      step = 0
      while step < num_iter and not coord.should_stop():
        pred_prob, gt_label, fc_fea = sess.run([prob_op, gt_op, fc_op])
        # print(gt_label)
        print(step)
        pred_prob_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = pred_prob
        gt_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = gt_label
        fc_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = fc_fea
        true_count += ((pred_prob > 0.5) == gt_label.astype('bool')).sum()

        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision = %.3f' % (datetime.now(), precision))

      # Compute mean averaged precision
      gt_all_bool = gt_all.astype('bool')
      aps = smetrics.average_precision_score(gt_all_bool, pred_prob_all, average=None)
      print(aps)
      meanAP = np.nanmean(aps)
      print('%s: mAP = %.3f' % (datetime.now(), meanAP))
      pred_metrics = dict()
      pred_metrics['gt'] = gt_all
      pred_metrics['pred'] = pred_prob_all
      pred_metrics['aps'] = aps
      pred_metrics['mAP'] = meanAP
      # sio.savemat(os.path.join(eval_dir, FLAGS.architecture + '_' + FLAGS.eval_data + '_lo-'+ str(FLAGS.leaveout) + '_pred.mat'), pred_metrics)
      sio.savemat(
        os.path.join(eval_dir, str(FLAGS.num_crops) + 'crops_' + FLAGS.architecture + '_' + FLAGS.eval_data + '_lo-' + str(FLAGS.leaveout) + '_pred.mat'),
        pred_metrics)

      fc_metrics = dict()
      fc_metrics['label'] = gt_all
      fc_metrics['feature'] = fc_all
      # sio.savemat(os.path.join(eval_dir, FLAGS.architecture + '_' + FLAGS.eval_data + '_lo-'+ str(FLAGS.leaveout) + '_fc.mat'), fc_metrics)
      sio.savemat(
        os.path.join(eval_dir, str(FLAGS.num_crops) + 'crops_' + FLAGS.architecture + '_' + FLAGS.eval_data + '_lo-' + str(FLAGS.leaveout) + '_fc.mat'),
        fc_metrics)
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
    # images, labels = model.inputs(eval_data=eval_data, leaveout=FLAGS.leaveout, batch_size=FLAGS.eval_batch_size, min_que=NUM_EXAMPLES[FLAGS.leaveout])
    images, labels = model.inputs(eval_data=eval_data, leaveout=FLAGS.leaveout, batch_size=FLAGS.eval_batch_size,
                                  min_que=NUM_EXAMPLES[FLAGS.leaveout], num_crops=FLAGS.num_crops)
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
      eval_once(saver, prob_op, labels, fcs)
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
