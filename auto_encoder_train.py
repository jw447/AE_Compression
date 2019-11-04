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

tf.app.flags.DEFINE_integer('max_steps', 10000,

                            """Number of steps to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_string('data_dir', './data/GROMACS_data/data_100step_1000seg',
                           """Path to GROMACS data directory: ./data/GROMACS_data/ """)

train_dir = FLAGS.train_dir
data_dir = FLAGS.data_dir

import auto_encoder_model as model

def train():
    # something wrong with feeding ------------------- June 27th 2018
    """Train Auto Encoder model for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # f = open('sedov-pres.dat','rb')
        f = open('data/GROMACS_data/data_100step_1000seg/md_0_seg.txt','r')
        input_data = f.readlines()

        for i in range(1,100):
            f = open('data/GROMACS_data/data_100step_1000seg/md_'+str(i)+'_seg.txt','r')
            input_data = np.concatenate([input_data,f.readlines()],0)

        input_data = [float(line.split(' ')[1]) for line in input_data]
        print(np.shape(input_data))
        input_data = np.reshape(input_data,[10000,1000])
        print(np.shape(input_data))

        inputs_ = tf.placeholder(tf.float32, (None,1000))
        # targets_ = tf.placeholder(tf.float32, (None,1000))
        
        representation, reconstruct = model.inference_fconn(inputs_)

        loss = model.loss(reconstruct, inputs_)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.

        # print(data)
        print(reconstruct)

        # # print(reconstruct)
        # images_reconstruct = tf.concat([targets_,reconstruct],1)
        # # print(images_reconstruct)
        # images_reconstruct = tf.reshape(images_reconstruct,[1,1000,2,1])
        # tf.summary.image('original_reconstruct', images_reconstruct, max_outputs=20)
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
            feed = {inputs_: input_data}
            loss_value, _ = sess.run([loss, train_op], feed_dict=feed)
            duration = time.time() - start_time
            
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:

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
