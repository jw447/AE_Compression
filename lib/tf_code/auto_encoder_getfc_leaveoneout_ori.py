# get feature representation from auto encoder, for original data

## python auto_encoder_getfc_leaveoneout_ori.py --leaveout 0

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


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../autoencoder_results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'ft_test',
                           """Either 'test' or 'train_eval' or 'ori_test' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../autoencoder_results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('leaveout', 0,
                           """number 0-12 .""")
tf.app.flags.DEFINE_integer('eval_batch_size', 1,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118')
tf.app.flags.DEFINE_string('data_dir', '../data/SyntheticScatteringData3BIN_log_32',
                           """Path to the CIFAR-10 data directory: ../data/SyntheticScatteringData3BIN_log_32, ../data/SyntheticScatteringData_multi_cropsBIN_log_32 """)

import auto_encoder_model as model

checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_tf_log_' + 'train_2_2_1024'
eval_dir = FLAGS.eval_dir +  FLAGS.data_type + '_tf_log_' + FLAGS.eval_data + '_2_2_1024'

print(FLAGS.leaveout)
NUM_CLASSES = model.NUM_CLASSES
NUM_EXAMPLES = [54, 618, 447, 91, 209, 134, 192, 26, 167, 54, 133, 244, 463]

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



def eval_once(saver, summary_writer, fc_op, gt_op, summary_op, images_op):
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

      gt_all = np.zeros([num_examples, NUM_CLASSES])
      # images_all = np.zeros([num_examples, 224,224,1])
      fc_dim = fc_op.get_shape()[1].value

      fc_all = np.zeros([num_examples, fc_dim])
      step = 0
      while step < num_iter and not coord.should_stop():
        # gt_label, fc_fea, images_ = sess.run([ gt_op, fc_op, images_op])
        gt_label, fc_fea = sess.run([gt_op, fc_op])
        # print(gt_label)
        print(step)
        gt_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = gt_label
        fc_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = fc_fea
        # images_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:,:,:] = images_
        step += 1


      fc_metrics = dict()
      fc_metrics['label'] = gt_all
      fc_metrics['feature'] = fc_all
      # fc_metrics['images'] = images_all
      sio.savemat(os.path.join(eval_dir,  'fc_label_lo-'+ str(FLAGS.leaveout) + '.mat'), fc_metrics)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
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
    images, labels = model.inputs(eval_data=eval_data, leaveout=FLAGS.leaveout, batch_size=FLAGS.eval_batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    representation, reconstruct = model.inference(images)

    # Calculate predictions.
    representation_reshape = tf.reshape(representation, [FLAGS.eval_batch_size, -1])

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    tf.image_summary('original', images, max_images=20)
    tf.image_summary('reconstruct', reconstruct, max_images=20)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(eval_dir, g)

    while True:
      eval_once(saver, summary_writer, representation_reshape, labels, summary_op, images)
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
