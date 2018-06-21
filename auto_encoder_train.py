# training auto encoder

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

tf.app.flags.DEFINE_string('train_dir', './autoencoder_results/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 25000,

                            """Number of steps to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# tf.app.flags.DEFINE_string('data_type', 'synthetic','data type, could be synthetic, ori, 3crops')
# tf.app.flags.DEFINE_integer('num_tags', 17,'17 or 118')
tf.app.flags.DEFINE_string('data_dir', './data/GROMACS_data/data_300step_1000seg',
                           """Path to GROMACS data directory: ./data/GROMACS_data/ """)

# train_dir = FLAGS.train_dir + FLAGS.data_type + '_tf_log_' + 'train_2_2_1024'
train_dir = FLAGS.train_dir
data_dir = FLAGS.data_dir

import auto_encoder_model as model

def train():
    """Train Auto Encoder model for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        data_obj = model.inputs()

        values = data_obj.value # list of 3 tensors with shape(300)
        # print(values)
        data =  tf.concat(values, 0)
        # print(data)
        # for conv2d - tensor has to be 4d
        # data = tf.reshape(data, [1, data_obj.height,data_obj.width, 1])

        # use 2d for now, with fc layer
        data = tf.reshape(data, [data_obj.height,data_obj.width])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # representation, reconstruct, dec4 = model.inference(images)
        representation, reconstruct = model.inference_fconn(data)

        loss = model.loss(tf.transpose(reconstruct), data)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.

        # print(data)
        # print(tf.transpose(reconstruct))


        images_reconstruct = tf.concat([data, tf.transpose(reconstruct)],1)

        images_reconstruct = tf.reshape(images_reconstruct,[1,1000,6,1])
        tf.summary.image('original_reconstruct', images_reconstruct, max_outputs=20)
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.log_device_placement=FLAGS.log_device_placement
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        # Start running operations on the Graph.
        sess = tf.Session(config=config)
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        # gradient_op = tf.gradients(loss, tf.all_variables())[0]
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # _, loss_value, gt_labels, gt_images, pred_logits = sess.run([train_op, loss, labels, images, logits])
            # _, loss_value, dec4_, rec_, rep_ = sess.run([train_op, loss, dec4, reconstruct, representation])
            _, loss_value, rec_, rep_ = sess.run([train_op, loss, reconstruct, representation])
            # loss_value = sess.run(loss)
            # grad_all_var = []
            # for grad_idx, var_idx in grad_op:
            #   if grad_idx is not None:
            #     grad_all_var.append(sess.run(grad_idx))
            duration = time.time() - start_time
            # assert not np.isnan(np.sum(gt_images)), 'Model diverged with images = NaN'
            # assert not np.isnan(np.sum(gt_labels)), 'Model diverged with gt_labels = NaN'
            # zeros_idx = np.nonzero(np.sum(1-gt_labels, 0))
            # if zeros_idx[0].shape[0] == 0:
            #   print(zeros_idx)
            # assert not np.isnan(np.sum(gt_labels)), 'Model diverged with gt_labels = NaN'
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            # if np.isnan(loss_value):
            #   print(pred_logits)
            #   print(gt_labels)

            if step % 10 == 0:
                # num_examples_per_step = FLAGS.batch_size
                # examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.3f sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, sec_per_batch))

            if step % 200 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    print('writing every thing to %s' % train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
