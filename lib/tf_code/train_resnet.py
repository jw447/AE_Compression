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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../results/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('architecture', 'resnet','network architecture, could be vgg_16 or alexnet')
tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118, 1, 2, 34')
tf.app.flags.DEFINE_integer('tag_id', 1,'0_16, exclude 0,5')
tf.app.flags.DEFINE_integer('tag_id0', -1,'0_16, exclude 0,5')
tf.app.flags.DEFINE_integer('tag_id1', -1,'0_16, exclude 0,5')
tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
tf.app.flags.DEFINE_float('memory', 0.4,'memory fraction')
tf.app.flags.DEFINE_integer('startep', -1,'starting epoch, 0 start from scratch')

import model_resnet as model
# train_dir = FLAGS.train_dir + 'tf_log_' + FLAGS.architecture + '_train'
# print(FLAGS.use_mag)
# train_dir = FLAGS.train_dir + 'tf_log_grad_mag-' + str(FLAGS.use_mag) + '_' + FLAGS.architecture + '_train'
if FLAGS.num_tags == 1:
  train_dir = FLAGS.train_dir + 'trainonall_'+ str(FLAGS.num_tags)+'tags_' + str(FLAGS.tag_id) + '_'+ FLAGS.architecture + '_train'
else:
  train_dir = FLAGS.train_dir + 'trainonall_' + str(FLAGS.num_tags) + 'tags_' + FLAGS.architecture + '_train'

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = model.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images, is_training=True)

    # Calculate loss.
    loss = model.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    config = tf.ConfigProto()
    config.log_device_placement=FLAGS.log_device_placement
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory
    # Start running operations on the Graph.
    sess = tf.Session(config=config)
    sess.run(init)

    step_init = 0

    if FLAGS.startep > 0:

      ckpt = tf.train.get_checkpoint_state(train_dir)

      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        print('load from pretrained model')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      else:
        print('No checkpoint file found')
        return

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    for step in np.arange(step_init, FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.startep <= 0:
    if tf.gfile.Exists(train_dir):
      tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  print('writing every thing to %s' % train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
