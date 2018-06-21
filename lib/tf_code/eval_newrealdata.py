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

from datetime import datetime
import math
import time
import sklearn.metrics as smetrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as sio
import numpy as np
import tensorflow as tf



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'new_realdata',
                           """Either 'test' or 'train_eval' or   'ori_test' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('eval_batch_size', 20,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('architecture', 'vgg_16','network architecture, could be vgg_16 or alexnet')
tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118')

checkpoint_dir = FLAGS.checkpoint_dir + 'tf_log_' + FLAGS.architecture + '_train'
eval_dir = FLAGS.eval_dir + 'tf_log_' + FLAGS.architecture + '_' + FLAGS.eval_data

import model
import newrealdata_loader as dataloader

NUM_CLASSES = model.NUM_CLASSES
NUM_DATA = dataloader.NUM_DATA

def eval_once(saver, summary_writer, prob_op, image_op):
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

      num_iter = int(math.ceil(NUM_DATA / FLAGS.eval_batch_size))
      print(NUM_DATA)
      pred_prob_all = np.zeros([NUM_DATA, NUM_CLASSES])
      loss = 0
      step = 0
      while step < num_iter and not coord.should_stop():

        eval_data = np.zeros([FLAGS.eval_batch_size, model.IMAGE_SIZE, model.IMAGE_SIZE, 1])
        eval_data[:min(FLAGS.eval_batch_size, NUM_DATA - step*FLAGS.eval_batch_size),:,:,:] = dataloader.DATA[step*FLAGS.eval_batch_size: min((step+1)*FLAGS.eval_batch_size, NUM_DATA), :, :, :]
        pred_prob = sess.run([prob_op], feed_dict={image_op:eval_data})
        pred_prob_all[step*FLAGS.eval_batch_size : min((step+1)*FLAGS.eval_batch_size, NUM_DATA),:] = pred_prob[0][:min(FLAGS.eval_batch_size, NUM_DATA-step*FLAGS.eval_batch_size) ,:]

        step += 1
        print(step)

      pred_metrics = dict()
      pred_metrics['pred'] = pred_prob_all
      sio.savemat(FLAGS.eval_data +'_'+ FLAGS.architecture +'_pred.mat', pred_metrics)


    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data
    images = tf.placeholder('float32', [FLAGS.eval_batch_size, model.IMAGE_SIZE, model.IMAGE_SIZE, 1])

    # images, labels = model.inputs(eval_data=eval_data, batch_size=FLAGS.eval_batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images, FLAGS.architecture)

    # Calculate predictions.
    prob_op = tf.sigmoid(logits)
    # top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(eval_dir, g)

    while True:
      eval_once(saver, summary_writer, prob_op, images)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()




