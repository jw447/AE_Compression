# get feature representation from auto encoder

## python auto_encoder_getfc.py --run_once True --eval_data ori_test --num_examples 2832 --architecture 5layer
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

# tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           # """Either 'test' or 'train_eval' or 'ori_test' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', './autoencoder_results/',
                           """Directory where to read model checkpoints.""")
# tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                             """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples', 80000,
#                             """Number of examples to run.""")
# tf.app.flags.DEFINE_integer('eval_batch_size', 50,
#                             """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

tf.app.flags.DEFINE_string('eval_dir', './autoencoder_results/eval/',
                           """Directory where to write event logs """)
# tf.app.flags.DEFINE_integer('max_steps', 25000,
                            # """Number of steps to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
# tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118')
tf.app.flags.DEFINE_string('data_dir', './data/GROMACS_data/data_300step_1000seg',
                           """Path to GROMACS data directory: ./data/GROMACS_data/ """)


import auto_encoder_model as model


# checkpoint_dir = FLAGS.checkpoint_dir + FLAGS.data_type + '_tf_log_' + 'train_2_2_1024'
# eval_dir = FLAGS.eval_dir +  FLAGS.data_type + '_tf_log_' + FLAGS.eval_data + '_2_2_1024'
checkpoint_dir = FLAGS.checkpoint_dir
eval_dir = FLAGS.eval_dir
# NUM_CLASSES = model.NUM_CLASSES

def eval_once(saver, summary_writer, fc_op, summary_op, reconstruct_op, images_op):
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

      # num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.eval_batch_size))
      # num_examples = num_iter * FLAGS.eval_batch_size
        fc_dim = fc_op.get_shape()[1].value

      # print(num_examples)
      # true_count = 0  # Counts the number of correct predictions.
      # gt_all = np.zeros([num_examples, NUM_CLASSES])
      # fc_all = np.zeros([num_examples, fc_dim])
      # step = 0
      # while step < num_iter and not coord.should_stop():
        image, recons = sess.run([images_op,reconstruct_op])
        # print(gt_label)
        # print(step)
        # gt_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = gt_label
        # fc_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = fc_fea

        # step += 1

      # Compute mean averaged precision

      fc_metrics = dict()
      fc_metrics['label'] = image
      fc_metrics['feature'] = recons
      sio.savemat(os.path.join(eval_dir, 'compare.mat'), fc_metrics)

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
    # eval_data = FLAGS.eval_data
    # images, labels = model.inputs(eval_data=eval_data, batch_size=FLAGS.eval_batch_size)
    data_obj = model.inputs()
    values = data_obj.value
    data = tf.concat(values,0)
    # print(data)

    data = tf.reshape(data, [data_obj.height,data_obj.width])
    # print(data)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    representation, reconstruct = model.inference_fconn(data)
    # print(representation)
    # print(reconstruct)
    # print(data)
    # Calculate predictions.
    # representation_reshape = tf.reshape(representation, [FLAGS.eval_batch_size, -1])

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # images_reconstruct = tf.concat([data, tf.transpose(reconstruct)],1)
    # print(images_reconstruct)
    # tf.summary.image('original_reconstruct', images_reconstruct)
    # tf.image_summary('original', images, max_images=20)
    # tf.image_summary('reconstruct', reconstruct, max_images=20)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)


    while True:
      eval_once(saver, summary_writer, representation, summary_op, reconstruct, data)
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
