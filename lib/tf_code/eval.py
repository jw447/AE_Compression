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
import os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval' or 'ori_test', 'test_orival' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run., 3200""")
tf.app.flags.DEFINE_integer('eval_batch_size', 40,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('architecture', 'vgg_16','network architecture, could be vgg_16 or alexnet')
tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118,1,2')
tf.app.flags.DEFINE_integer('tag_id', 1,'0_16, exclude 0,5')
tf.app.flags.DEFINE_integer('tag_id0', -1,'0_16, exclude 0,5')
tf.app.flags.DEFINE_integer('tag_id1', -1,'0_16, exclude 0,5')
tf.app.flags.DEFINE_boolean('plot_roc', False, 'whether to plot')
tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
tf.app.flags.DEFINE_boolean('weighted', False,'whether to weight the loss')
tf.app.flags.DEFINE_string('data_dir', '../data/SyntheticScatteringData3BIN_log_32',
                           """Path to the CIFAR-10 data directory: ../data/SyntheticScatteringData3BIN_log_32, ../data/SyntheticScatteringData_multi_cropsBIN_log_32 """)
tf.app.flags.DEFINE_boolean('save_image', False,'whether to save the image')
# checkpoint_dir = FLAGS.checkpoint_dir + '3crops2_tf_log_' + str(FLAGS.num_tags)+'tags_' + FLAGS.architecture + '_train'
# eval_dir = FLAGS.eval_dir + '3crops2_tf_log_' + str(FLAGS.num_tags)+'tags_' + FLAGS.architecture + '_' + FLAGS.eval_data


# checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_tf_log_' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_train'
# eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_tf_log_' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_' + FLAGS.eval_data

# checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_tf_log_linearbs' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
# eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_tf_log_linearbs' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_' + FLAGS.eval_data

# checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_tf_log_polycrl' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
# eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_tf_log_polycrl' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_' + FLAGS.eval_data

# checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_small_tf_log_' + str(FLAGS.num_tags)+'tags2_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
# eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_small_tf_log_' + str(FLAGS.num_tags)+'tags2_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_' + FLAGS.eval_data

# train_dir = FLAGS.train_dir + FLAGS.data_type + '_small_'+ str(FLAGS.tag_id) + '_' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
if FLAGS.num_tags == 2:
  checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_small_'+ str(FLAGS.num_tags)+'tags_' + str(FLAGS.tag_id0) + '_'+ str(FLAGS.tag_id1) + '_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
  eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_small_'+ str(FLAGS.num_tags)+'tags_' + str(FLAGS.tag_id0) + '_'+ str(FLAGS.tag_id1) + '_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_' + FLAGS.eval_data
elif FLAGS.num_tags == 17:
  checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_all_'+ str(FLAGS.tag_id) + '_' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
  eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_all_'+ str(FLAGS.tag_id) + '_' + str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_' + FLAGS.eval_data

  # checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_small_' + str(FLAGS.tag_id) + '_' + str(
  #   FLAGS.num_tags) + 'tags_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_train'
  # eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_small_' + str(FLAGS.tag_id) + '_' + str(
  #   FLAGS.num_tags) + 'tags_weighted-' + str(FLAGS.weighted) + '_' + FLAGS.architecture + '_' + FLAGS.eval_data

elif FLAGS.num_tags == 1:
  checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_small_' + str(FLAGS.tag_id)+ '_'+ str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_train'
  eval_dir = FLAGS.eval_dir + FLAGS.data_type + '_small_' + str(FLAGS.tag_id)+ '_'+ str(FLAGS.num_tags)+'tags_weighted-' + str(FLAGS.weighted) + '_'+ FLAGS.architecture + '_' + FLAGS.eval_data


print(checkpoint_dir)
print(eval_dir)

import model


NUM_CLASSES = model.NUM_CLASSES

def eval_once(saver, summary_writer, prob_op, images_op, gt_op, summary_op):
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

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.eval_batch_size))
      num_examples = num_iter * FLAGS.eval_batch_size
      print(num_examples)
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_examples * NUM_CLASSES
      pred_prob_all = np.zeros([num_examples, NUM_CLASSES])
      gt_all = np.zeros([num_examples, NUM_CLASSES])
      im_height = images_op.get_shape()[1].value
      im_width = images_op.get_shape()[2].value
      images_all = np.zeros([num_examples, im_height, im_width])
      loss = 0
      step = 0
      while step < num_iter and not coord.should_stop():
        print(step)
        pred_prob, gt_label, images_ = sess.run([prob_op, gt_op, images_op])

        pred_prob_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = pred_prob
        gt_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = gt_label
        # print(images_.shape)
        images_all[step * FLAGS.eval_batch_size: (step + 1) * FLAGS.eval_batch_size, :,:] = images_[:,:,:,0]
        true_count += ((pred_prob > 0.5) == gt_label.astype('bool')).sum()

        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision = %.3f' % (datetime.now(), precision))
      print('num positive: %d' % np.sum(gt_all))
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
      # sio.savemat(FLAGS.architecture +'_pred.mat', pred_metrics)
      sio.savemat(os.path.join(eval_dir,  FLAGS.eval_data+'_pred.mat'), pred_metrics)

      if FLAGS.save_image:
        image_toSave = dict()
        image_toSave['images'] = images_all
        sio.savemat(os.path.join(eval_dir, FLAGS.eval_data+'_images.mat'), image_toSave)

      # plot ROC curve
      if FLAGS.plot_roc:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_idx in range(NUM_CLASSES):
          fpr[class_idx], tpr[class_idx], _ = roc_curve(gt_all[:,class_idx], pred_prob_all[:,class_idx])
          roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
          plt.figure()
          plt.plot(fpr[class_idx], tpr[class_idx], label='ROC curve (area = %0.2f)' % roc_auc[class_idx])
          plt.plot([0, 1], [0, 1], 'k--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.title('Receiver operating characteristic class %d'% class_idx)
          plt.legend(loc="lower right")
          plt.savefig('roc_%d.pdf' % class_idx)
          plt.savefig('roc_%d.png' % class_idx)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision', simple_value=precision)
      summary.value.add(tag='mAP', simple_value=meanAP)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data
    images, labels = model.inputs(eval_data=eval_data, batch_size=FLAGS.eval_batch_size)

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
      eval_once(saver, summary_writer, prob_op, images,  labels, summary_op)
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
