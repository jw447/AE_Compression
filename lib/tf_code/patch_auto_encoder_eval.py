# read the resized image from h5 files
# densely getting patches, get the corresponding representation vector
# perform spatial pyramid pooling
# save final feature vectors for each image, for future LSSVM

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
import ML_HDF5
from Get_Patches import Get_Patches
from ML_PyrPooling import pyrPool

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../patch_autoencoder_results/',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../patch_autoencoder_results/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_integer('eval_batch_size', 500,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('data_dir', '../data/ori_patches',
                           """Path to the CIFAR-10 data directory: ../data/ori_patches, ../data/SyntheticScatteringData3BIN_log_32, ../data/SyntheticScatteringData_multi_cropsBIN_log_32 """)
tf.app.flags.DEFINE_integer('img_start', 0,
                            """Number of batch size to evaluate.""")
tf.app.flags.DEFINE_integer('img_end', 10,
                            """Number of batch size to evaluate.""")



import patch_auto_encoder_model as model

checkpoint_dir = FLAGS.checkpoint_dir + 'tf_log_' + 'train_int32_1_1_1024'
eval_dir = FLAGS.eval_dir +  'tf_log_' + 'eval_int32_1_1_1024'

#
# def eval_once(saver, summary_writer, fc_op, gt_op, summary_op, reconstruct_op, images_op):
#     """Run Eval once.
#
#     Args:
#       saver: Saver.
#       summary_writer: Summary writer.
#       prob_op: prob op.
#       gt_op: ground truth op.
#       summary_op: Summary op.
#     """
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     # config.gpu_options.per_process_gpu_memory_fraction = 0.3
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
#         # Start the queue runners.
#         coord = tf.train.Coordinator()
#         try:
#             threads = []
#             for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#                 threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                                  start=True))
#
#             num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.eval_batch_size))
#             num_examples = num_iter * FLAGS.eval_batch_size
#             fc_dim = fc_op.get_shape()[1].value
#
#             print(num_examples)
#             true_count = 0  # Counts the number of correct predictions.
#             gt_all = np.zeros([num_examples, NUM_CLASSES])
#             fc_all = np.zeros([num_examples, fc_dim])
#             step = 0
#             while step < num_iter and not coord.should_stop():
#                 gt_label, fc_fea = sess.run([ gt_op, fc_op])
#                 # print(gt_label)
#                 print(step)
#                 gt_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = gt_label
#                 fc_all[step*FLAGS.eval_batch_size : (step+1)*FLAGS.eval_batch_size,:] = fc_fea
#
#                 step += 1
#
#             # Compute mean averaged precision
#
#             fc_metrics = dict()
#             fc_metrics['label'] = gt_all
#             fc_metrics['feature'] = fc_all
#             sio.savemat(os.path.join(eval_dir, 'fc_label.mat'), fc_metrics)
#
#             summary = tf.Summary()
#             summary.ParseFromString(sess.run(summary_op))
#
#             summary_writer.add_summary(summary, global_step)
#
#         except Exception as e:  # pylint: disable=broad-except
#             coord.request_stop(e)
#
#         coord.request_stop()
#         coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    # read data
    data_file_name = '../data/ori_patches/2_ori_resized_img_int32.h5'

    PATCH_SIZE = 8*4
    STEP = 2*4
    RESIZE = np.array([16, 32, 48, 64, 80, 96, 112, 128, 144])
    RESIZE = RESIZE * 4
    NUM_PATCHES = 0
    for a in range(len(RESIZE)):
        NUM_PATCHES = NUM_PATCHES + ((RESIZE[a] - PATCH_SIZE)/STEP + 1)*((RESIZE[a] - PATCH_SIZE)/STEP + 1)

    print('number of patches per image: %d' % NUM_PATCHES)

    print('loading pretrained auto encoder')
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, [FLAGS.eval_batch_size, PATCH_SIZE * PATCH_SIZE])
        images_list = tf.unpack(images)
        images_reshape_list = []
        for i in range(FLAGS.eval_batch_size):
            images_reshape_list.append( tf.image.per_image_whitening(tf.reshape(images_list[i], [PATCH_SIZE, PATCH_SIZE, 1])) )
        float_image = tf.pack(images_reshape_list)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        representation_bf,  representation_af, reconstruct = model.inference(float_image)
        rep_dim = representation_af.get_shape()[1].value

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        images_reconstruct = tf.concat(2, [float_image, reconstruct] )
        tf.image_summary('original_reconstruct', images_reconstruct, max_images=50)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(eval_dir, g)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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


            print('loading data')
            resized_img = ML_HDF5.load(data_file_name)
            print('loding data done')
            num_scale = len(resized_img)
            num_img = resized_img[0].shape[1]
            img_pooled_fea = []

            # for img_idx in range(num_img):
            for img_idx in range(FLAGS.img_start, FLAGS.img_end):
                print('processing img %d'%img_idx)
                patches_img = np.zeros([PATCH_SIZE*PATCH_SIZE, NUM_PATCHES])
                patches_img_locs = np.zeros([2, NUM_PATCHES])
                patches_img_cnt = 0
                fea_img = np.zeros([rep_dim, NUM_PATCHES])
                for scale_idx in range(num_scale):
                    foo_img = resized_img[scale_idx][:,img_idx]
                    batch_patches, batch_patches_locs = Get_Patches(foo_img, PATCH_SIZE, STEP)
                    # shift location of patches to center of patches
                    batch_patches_locs = batch_patches_locs + np.int((PATCH_SIZE-1)/2)
                    batch_patches_locs = batch_patches_locs.astype('double') / RESIZE[scale_idx]

                    num_patches_for_resize = batch_patches.shape[1]
                    patches_img[:, patches_img_cnt:patches_img_cnt+num_patches_for_resize] = batch_patches
                    patches_img_locs[:, patches_img_cnt:patches_img_cnt+num_patches_for_resize] = batch_patches_locs
                    patches_img_cnt += num_patches_for_resize

                # use auto encoder to get representation
                num_iter = np.ceil(NUM_PATCHES / (FLAGS.eval_batch_size*1.0)).astype('int')
                for step in range(num_iter):
                    print(step)
                    foo_patch = np.zeros([PATCH_SIZE*PATCH_SIZE, FLAGS.eval_batch_size])
                    foo_patch[:, :min(FLAGS.eval_batch_size, NUM_PATCHES - step*FLAGS.eval_batch_size)] = patches_img[:, step*FLAGS.eval_batch_size:min((step+1)*FLAGS.eval_batch_size, NUM_PATCHES)]
                    foo_rep = sess.run(representation_af, feed_dict={images:foo_patch.transpose()})
                    fea_img[:,step*FLAGS.eval_batch_size:min((step+1)*FLAGS.eval_batch_size, NUM_PATCHES)] = foo_rep[:min(FLAGS.eval_batch_size, NUM_PATCHES - step*FLAGS.eval_batch_size),:].transpose()

                # pyramid pooling
                img_pooled_fea_foo = pyrPool(patches_img_locs, fea_img, np.array([[0,1],[0,1]]), 'sum', np.array([1/4, 1/2, 1]))
                img_pooled_fea.append(img_pooled_fea_foo)
            # img_pooled_fea = np.array(img_pooled_fea)
            print('saving features')
            ML_HDF5.save(os.path.join(eval_dir, '2_pooled_fea_%d_%d.h5' %(FLAGS.img_start, FLAGS.img_end)), img_pooled_fea, np.array([FLAGS.img_end - FLAGS.img_start]))
            print('saving done')


def get_some_ori_reconstruct_img():
    # read data
    data_file_name = '../data/ori_patches/2_ori_resized_img_int32.h5'

    PATCH_SIZE = 8 * 4
    STEP = 2 * 4
    RESIZE = np.array([16, 32, 48, 64, 80, 96, 112, 128, 144])
    RESIZE = RESIZE * 4
    NUM_PATCHES = 0
    for a in range(len(RESIZE)):
        NUM_PATCHES = NUM_PATCHES + ((RESIZE[a] - PATCH_SIZE) / STEP + 1) * ((RESIZE[a] - PATCH_SIZE) / STEP + 1)

    print('number of patches per image: %d' % NUM_PATCHES)

    print('loading pretrained auto encoder')
    with tf.Graph().as_default() as g:
        images = tf.placeholder(tf.float32, [FLAGS.eval_batch_size, PATCH_SIZE * PATCH_SIZE])
        images_list = tf.unpack(images)
        images_reshape_list = []
        for i in range(FLAGS.eval_batch_size):
            images_reshape_list.append(
                tf.image.per_image_whitening(tf.reshape(images_list[i], [PATCH_SIZE, PATCH_SIZE, 1])))
        float_image = tf.pack(images_reshape_list)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        representation_bf, representation_af, reconstruct = model.inference(float_image)
        rep_dim = representation_af.get_shape()[1].value

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        images_reconstruct = tf.concat(2, [float_image, reconstruct])
        tf.image_summary('original_reconstruct', images_reconstruct, max_images=50)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(eval_dir, g)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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

            print('loading data')
            resized_img = ML_HDF5.load(data_file_name)
            print('loding data done')
            num_scale = len(resized_img)
            img_ori_patches = []
            img_reconst_patches = []

            # for img_idx in range(num_img):
            num_img_to_reconst = 10
            NUM_PATCHES_to_save = 100
            img_idx_to_reconst = np.random.randint(0, 2832, [num_img_to_reconst])
            for img_idx_i in range(num_img_to_reconst):
                img_idx = img_idx_to_reconst[img_idx_i]
                print('processing img %d' % img_idx)
                patches_img = np.zeros([PATCH_SIZE * PATCH_SIZE, NUM_PATCHES])
                patches_img_locs = np.zeros([2, NUM_PATCHES])
                patches_img_cnt = 0
                fea_img = np.zeros([rep_dim, NUM_PATCHES])
                tmp_ori_patch_to_save = np.zeros([PATCH_SIZE*PATCH_SIZE, NUM_PATCHES_to_save])
                tmp_reconst_patch_to_save = np.zeros([PATCH_SIZE*PATCH_SIZE, NUM_PATCHES_to_save])

                for scale_idx in range(num_scale):
                    foo_img = resized_img[scale_idx][:, img_idx]
                    batch_patches, batch_patches_locs = Get_Patches(foo_img, PATCH_SIZE, STEP)
                    # shift location of patches to center of patches
                    batch_patches_locs = batch_patches_locs + np.int((PATCH_SIZE - 1) / 2)
                    batch_patches_locs = batch_patches_locs.astype('double') / RESIZE[scale_idx]

                    num_patches_for_resize = batch_patches.shape[1]
                    patches_img[:, patches_img_cnt:patches_img_cnt + num_patches_for_resize] = batch_patches
                    patches_img_locs[:, patches_img_cnt:patches_img_cnt + num_patches_for_resize] = batch_patches_locs
                    patches_img_cnt += num_patches_for_resize

                rdm_patch_idx = np.random.randint(0, NUM_PATCHES, [NUM_PATCHES_to_save])
                tmp_ori_patch_to_save = patches_img[:, rdm_patch_idx]
                # use auto encoder to get representation
                num_iter = np.ceil(NUM_PATCHES_to_save / (FLAGS.eval_batch_size * 1.0)).astype('int')
                for step in range(num_iter):
                    print(step)
                    foo_patch = np.zeros([PATCH_SIZE * PATCH_SIZE, FLAGS.eval_batch_size])
                    foo_patch[:, :min(FLAGS.eval_batch_size, NUM_PATCHES_to_save - step * FLAGS.eval_batch_size)] = tmp_ori_patch_to_save[:,step * FLAGS.eval_batch_size:min((step + 1) * FLAGS.eval_batch_size,NUM_PATCHES_to_save)]
                    foo_reconst = sess.run(reconstruct, feed_dict={images: foo_patch.transpose()})
                    foo_reconst = np.transpose(np.reshape(foo_reconst, [foo_reconst.shape[0], -1]))
                    tmp_reconst_patch_to_save[:,
                    step * FLAGS.eval_batch_size:min((step + 1) * FLAGS.eval_batch_size, NUM_PATCHES_to_save)] = foo_reconst[:, :min(
                        FLAGS.eval_batch_size, NUM_PATCHES - step * FLAGS.eval_batch_size)]


                img_ori_patches.append(tmp_ori_patch_to_save)
                img_reconst_patches.append(tmp_reconst_patch_to_save)

            print('saving features')
            ML_HDF5.save(os.path.join(eval_dir, 'ori_patches.h5'),img_ori_patches, np.array([num_img_to_reconst, 1]))
            ML_HDF5.save(os.path.join(eval_dir, 'reconst_patches.h5'), img_reconst_patches,
                         np.array([num_img_to_reconst,1]))
            print('saving done')


def get_cluser_img():
    num_cluster = 1024
    print('loading pretrained auto encoder')
    with tf.Graph().as_default() as g:
        representation_af = tf.placeholder(tf.float32, [FLAGS.eval_batch_size, num_cluster])
        # Build a Graph that computes the logits predictions from the
        # inference model.
        reconstruct = model.get_cluster(representation_af, is_training=False)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        tf.image_summary('original_reconstruct', reconstruct, max_images=50)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(eval_dir, g)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

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

            rep_all = np.eye(num_cluster)
            num_iter = np.ceil(num_cluster / (FLAGS.eval_batch_size * 1.0)).astype('int')
            img_cluster = np.zeros([32*32, num_cluster])
            for step in range(num_iter):
                print(step)

                foo_rep = np.zeros([num_cluster, FLAGS.eval_batch_size])
                foo_rep[:, :min(FLAGS.eval_batch_size,
                                  num_cluster - step * FLAGS.eval_batch_size)] = rep_all[:,
                                                                                         step * FLAGS.eval_batch_size:min(
                                                                                             (
                                                                                             step + 1) * FLAGS.eval_batch_size,
                                                                                             num_cluster)]
                foo_reconst = sess.run(reconstruct, feed_dict={representation_af: foo_rep.transpose()})
                foo_reconst = np.transpose(np.reshape(foo_reconst, [foo_reconst.shape[0], -1]))
                img_cluster[:,
                step * FLAGS.eval_batch_size:min((step + 1) * FLAGS.eval_batch_size,
                                                 num_cluster)] = foo_reconst[:, :min(
                    FLAGS.eval_batch_size, num_cluster - step * FLAGS.eval_batch_size)]

            print('saving clusters')
            img_cluster_to_save = {}
            img_cluster_to_save['cluster'] = img_cluster
            sio.savemat(os.path.join(eval_dir, 'cluster.mat'), img_cluster_to_save )
            print('saving done')

def main(argv=None):  # pylint: disable=unused-argument
    # if tf.gfile.Exists(eval_dir):
    #     tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    # evaluate()
    # get_some_ori_reconstruct_img()
    get_cluser_img()

if __name__ == '__main__':
    tf.app.run()
