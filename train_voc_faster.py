#coding=utf-8
import tensorflow as tf
from losses.ret_nms_loss import get_loss
from dsl_data import visual
import config
from tensorflow.contrib import slim
from utils import np_utils
import glob
import cv2
import numpy as np
import time
from data_set import data_gen
from dsl_data.utils import resize_image_fixed_size

import config
from models import fpn_faster_rcnn
def train():
    pl_images = tf.placeholder(shape=[config.batch_size, config.image_size[0], config.image_size[1], 3], dtype=tf.float32)
    pl_gt_boxs = tf.placeholder(shape=[config.batch_size, 100, 4], dtype=tf.float32)
    pl_label = tf.placeholder(shape=[config.batch_size, 100], dtype=tf.int32)
    pl_input_rpn_match = tf.placeholder(shape=[config.batch_size, config.total_anchor_num], dtype=tf.int32)
    pl_input_rpn_bbox = tf.placeholder(shape=[config.batch_size, config.total_anchor_num, 4], dtype=tf.float32)

    train_tensors, ta_gt = fpn_faster_rcnn.get_train_tensor(pl_images,  pl_input_rpn_match,pl_input_rpn_bbox, pl_label,pl_gt_boxs)

    gen_bdd = data_gen.get_batch(batch_size=config.batch_size, class_name='voc', image_size=config.image_size,
                                 max_detect=100,is_rcnn=True)

    global_step = slim.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=400000,
        decay_rate=0.9,
        staircase=True)

    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)
    vbs = []
    for s in slim.get_variables():
        print(s.name)
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name and 'GroupNorm' not in s.name:
            print(s.name)
            vbs.append(s)
    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, config.check_dir)

    sv = tf.train.Supervisor(logdir=config.save_dir, summary_op=None, init_fn=restore)


    with sv.managed_session() as sess:

        for step in range(20000000):
            print('       ' + ' '.join(['*'] * (step % 10)))

            images, true_box, true_label = next(gen_bdd)


            rpn_label, rpn_box = np_utils.build_rpn_targets(true_box, true_label, batch_size=config.batch_size,
                                                        cfg=config.Config)

            t1 = np.zeros(shape=true_box.shape, dtype=np.float32)
            t2 = np.zeros(shape=rpn_box.shape, dtype=np.float32)

            t1[:, : ,0 ] = true_box[:,:,1]
            t1[:, :, 1] = true_box[:, :, 0]
            t1[:, :, 2] = true_box[:, :, 3]
            t1[:, :, 3] = true_box[:, :, 2]

            t2[:, :, 0] = rpn_box[:, :, 1]
            t2[:, :, 1] = rpn_box[:, :, 0]
            t2[:, :, 2] = rpn_box[:, :, 3]
            t2[:, :, 3] = rpn_box[:, :, 2]


            feed_dict = {pl_images: images, pl_gt_boxs: t1,
                         pl_label: true_label, pl_input_rpn_bbox: t2,
                         pl_input_rpn_match: rpn_label}
            ls, step,tg = sess.run([train_op, global_step,ta_gt], feed_dict=feed_dict)

            if step % 10 == 0:
                print('step:' + str(step) +
                      ' ' + 'rpn_class_loss:' + str(ls[0]) +
                      ' ' + 'rpn_loc_loss:' + str(ls[1])+
                      ' ' + 'class_loss:' + str(ls[2])+
                      ' ' + 'loc_loss:' + str(ls[3])
                      )
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)


def detect():
    config.batch_size = 1
    ig = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)
    wind = tf.placeholder(shape=(4, 1), dtype=tf.float32)
    detections = fpn_faster_rcnn.predict(images=ig, window=wind)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/obj_detect/faster/model.ckpt-16505')
        for ip in glob.glob(
                '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit/VOC2007/JPEGImages/*.jpg'):
            print(ip)
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = resize_image_fixed_size(img,config.image_size)
            window = np.asarray(window) / config.image_size[0] * 1.0

            window = np.reshape(window, [4, 1])

            img = org - [123.15, 115.90, 103.06]
            img = np.expand_dims(img, axis=0)
            t = time.time()
            detects = sess.run([detections], feed_dict={ig: img, wind: window})
            arr = detects[0]
            ix = np.where(np.sum(arr, axis=1) > 0)
            box = arr[ix]
            boxes = box[:, 0:4]
            label = box[:, 4]
            score = box[:, 5]
            label = np.asarray(label, np.int32)-1
            visual.display_instances_title(org, np.asarray(boxes) * 512, class_ids=label,
                                           class_names=config.VOC_CLASSES, scores=score, is_faster=True)







train()