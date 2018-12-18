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
from faster_rcnn_config import config_instace as cfg
import config
from dsl_data import data_loader_multi
from models import light_head
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def train():
    pl_images = tf.placeholder(shape=[config.batch_size, config.image_size[0], config.image_size[1], 3], dtype=tf.float32)
    pl_gt_boxs = tf.placeholder(shape=[config.batch_size, 100, 4], dtype=tf.float32)
    pl_label = tf.placeholder(shape=[config.batch_size, 100], dtype=tf.int32)
    pl_input_rpn_match = tf.placeholder(shape=[config.batch_size, config.total_anchor_num], dtype=tf.int32)
    pl_input_rpn_bbox = tf.placeholder(shape=[config.batch_size, config.total_anchor_num, 4], dtype=tf.float32)

    train_tensors, ta_gt, ids = light_head.get_train_tensor(pl_images,  pl_input_rpn_match,pl_input_rpn_bbox, pl_label,pl_gt_boxs)

    gen_bdd = data_gen.get_batch(batch_size=config.batch_size, class_name='guoshu', image_size=config.image_size,
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
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name and 'GroupNorm' not in s.name:

            vbs.append(s)
    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, config.check_dir)

    sv = tf.train.Supervisor(logdir=config.save_dir, summary_op=None, init_fn=restore)


    with sv.managed_session() as sess:

        for step in range(20000000):
            print('       ' + ' '.join(['*'] * (step % 10)))

            images, true_box, true_label = next(gen_bdd)


            rpn_label, rpn_box = np_utils.build_rpn_targets_light_head(true_box, true_label, batch_size=config.batch_size,
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
            ls, step,tg, idx = sess.run([train_op, global_step,ta_gt,ids], feed_dict=feed_dict)
            #print(idx)
            if step % 10 == 0:
                print('step:' + str(step) +
                      ' ' + 'rpn_class_loss:' + str(tg[0]) +
                      ' ' + 'rpn_loc_loss:' + str(tg[1])+
                      ' ' + 'class_loss:' + str(tg[2])+
                      ' ' + 'bbox_loss:' + str(tg[3])+
                      ' ' + 'total_loss:' + str(ls)
                      )
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)


def detect():
    config.batch_size = 1
    cfg.NMS_ROIS_TRAINING = 1000
    ig = tf.placeholder(shape=(1, config.image_size[0], config.image_size[1], 3), dtype=tf.float32)
    wind = tf.placeholder(shape=(4, 1), dtype=tf.float32)
    detections,pp = light_head.predict1(images=ig, window=wind)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/obj_detect/guoshu_light_head_align/model.ckpt-2521')
        for ip in glob.glob(
                '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/*/*.png'):
            print(ip)
            img = cv2.imread(ip)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = resize_image_fixed_size(img,config.image_size)
            window = np.asarray(window) / config.image_size[0] * 1.0

            window = np.reshape(window, [4, 1])

            img = org - [123.15, 115.90, 103.06]
            img = np.expand_dims(img, axis=0)
            t = time.time()
            detects,p = sess.run([detections,pp], feed_dict={ig: img, wind: window})
            #arr = detects[0]
            print(p)
            print(detects)
            ix = np.where(np.sum(detects, axis=1) > 0)

            box = detects[ix]
            boxes = box[:, 0:4]
            label = box[:, 4]
            score = box[:, 5]
            label = np.asarray(label, np.int32)-1
            visual.display_instances_title(org, np.asarray(boxes) * 256, class_ids=label,
                                           class_names=config.VOC_CLASSES, scores=score, is_faster=True)
def detect1():
    config.batch_size = 1
    cfg.NMS_ROIS_TRAINING = 1000
    ig = tf.placeholder(shape=(1, config.image_size[0], config.image_size[1], 3), dtype=tf.float32)
    wind = tf.placeholder(shape=(4, 1), dtype=tf.float32)
    bbox, scores, classes = light_head.predict(images=ig, window=wind)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/obj_detect/guoshu_ll/model.ckpt-37710')
        for ip in glob.glob(
                '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/*/*.png'):
            print(ip)
            img = cv2.imread(ip)
            imges = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = resize_image_fixed_size(imges,config.image_size)
            window = np.asarray(window) / config.image_size[0] * 1.0

            window = np.reshape(window, [4, 1])

            img = org - [123.15, 115.90, 103.06]
            img = np.expand_dims(img, axis=0)
            t = time.time()
            p_box, p_score, p_class = sess.run([bbox, scores, classes], feed_dict={ig: img, wind: window})
            print(p_box)
            if p_box.shape[0] > 0:
                #bxx = np.asarray(bxx)*np.asarray([config.image_size[1],config.image_size[0],config.image_size[1],config.image_size[0]])
                #visual.display_instances_title(org, np.asarray(bxx), class_ids=np.asarray(cls),class_names=config.VOC_CLASSES,scores=scores)
                bxx = np_utils.revert_image(scale, padding, config.image_size, p_box)
                visual.display_instances_title(imges, bxx, class_ids=np.asarray(p_class),class_names=config.Lvcai,scores=p_score)


def tt():
    gen_bdd = data_gen.get_batch(batch_size=config.batch_size, class_name='voc', image_size=config.image_size,
                                 max_detect=100, is_rcnn=True)

    for step in range(20000000):
        print('       ' + ' '.join(['*'] * (step % 10)))

        images, true_box, true_label = next(gen_bdd)
        print(np.where(true_label>0))
        rpn_label, rpn_box = np_utils.build_rpn_targets(true_box, true_label, batch_size=config.batch_size,
                                                        cfg=config.Config)
        print(np.where(rpn_label==1))



detect()