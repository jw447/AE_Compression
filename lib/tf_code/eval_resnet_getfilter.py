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

import model_resnet as model
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval' or 'ori_test' .""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('eval_batch_size', 1,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('architecture', 'resnet','network architecture, could be vgg_16 or alexnet')
tf.app.flags.DEFINE_integer('gpu', 1, 'gpuid')

checkpoint_dir = FLAGS.checkpoint_dir + 'tf_log_' + FLAGS.architecture + '_train'
eval_dir = FLAGS.eval_dir + 'tf_log_' + FLAGS.architecture + '_' + FLAGS.eval_data

NUM_CLASSES = model.NUM_CLASSES

def eval_once(saver, summary_writer, prob_op, gt_op, summary_op):
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



        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    file_name = checkpoint_dir + '/model.ckpt-947500'

    try:
        reader = tf.train.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            keys = [i for i in var_to_shape_map]
            sorted(keys)

            for key in keys:
                print("tensor_name: ", key)
                # print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    # with tf.Graph().as_default() as g:
    #     # Get images and labels for CIFAR-10.
    #     eval_data = FLAGS.eval_data
    #     images, labels = model.inputs(eval_data=eval_data, batch_size=FLAGS.eval_batch_size)
    #
    #     # Build a Graph that computes the logits predictions from the
    #     # inference model.
    #     logits = model.inference(images, is_training=False)
    #
    #     # Calculate predictions.
    #     prob_op = tf.sigmoid(logits)
    #
    #     saver = tf.train.Saver(tf.all_variables())
    #
    #     # Build the summary operation based on the TF collection of Summaries.
    #     summary_op = tf.merge_all_summaries()
    #
    #     summary_writer = tf.train.SummaryWriter(eval_dir, g)
    #
    #     with tf.Session(config=config) as sess:
    #         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #         if ckpt and ckpt.model_checkpoint_path:
    #             # Restores from checkpoint
    #             saver.restore(sess, ckpt.model_checkpoint_path)
    #             # Assuming model_checkpoint_path looks something like:
    #             #   /my-favorite-path/cifar10_train/model.ckpt-0,
    #             # extract global_step from it.
    #             global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #         else:
    #             print('No checkpoint file found')
    #             return
    #
    #         print('loading done')

def main(argv=None):  # pylint: disable=unused-argument
    # if tf.gfile.Exists(eval_dir):
    #   tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
