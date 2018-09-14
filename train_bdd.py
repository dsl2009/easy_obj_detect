#coding=utf-8
import tensorflow as tf
from loss import get_loss
from dsl_data import visual
import config
from model import get_box_logits,predict
from dsl_data.utils import resize_image
from tensorflow.contrib import slim
import np_utils
import glob
import cv2
import numpy as np
import time
import data_gen

def train():
    img = tf.placeholder(shape=[config.batch_size, config.image_size[0], config.image_size[1], 3], dtype=tf.float32)
    loc = tf.placeholder(shape=[config.batch_size, config.total_anchor_num, 4], dtype=tf.float32)
    conf = tf.placeholder(shape=[config.batch_size, config.total_anchor_num], dtype=tf.float32)
    print(config.total_anchor_num)
    pred_loc, pred_confs, vbs = get_box_logits(img,config)

    train_tensors = get_loss(conf, loc, pred_loc, pred_confs,config)
    gen = data_gen.get_batch(batch_size=config.batch_size,class_name='bdd',image_size=config.image_size,max_detect=80)

    global_step = slim.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=40000,
        decay_rate=0.7,
        staircase=True)

    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
    train_op = slim.learning.create_train_op(train_tensors, optimizer)
    vbs = []
    for s in slim.get_variables():
        print(s.name)
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name:
            print(s.name)
            vbs.append(s)
    saver = tf.train.Saver(vbs)

    def restore(sess):
        saver.restore(sess, config.check_dir)


    sv = tf.train.Supervisor(logdir=config.save_dir, summary_op=None, init_fn=restore)

    with sv.managed_session() as sess:
        for step in range(200000):
            print('       '+' '.join(['*']*(step%10)))
            images, true_box, true_label = next(gen)

            loct, conft = np_utils.get_loc_conf(true_box, true_label, batch_size=config.batch_size,cfg=config.Config)
            feed_dict = {img: images, loc: loct,
                         conf: conft}

            ls, step = sess.run([train_op, global_step], feed_dict=feed_dict)
            if step % 10 == 0:
                print('step:' + str(step) +
                      ' ' + 'class_loss:' + str(ls[0]) +
                      ' ' + 'loc_loss:' + str(ls[1])
                      )
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)

def detect():
    config.batch_size = 1
    imgs = tf.placeholder(shape=(1, config.image_size, config.image_size, 3), dtype=tf.float32)
    #ig = AddCoords(x_dim=512, y_dim=512)(imgs)
    pred_loc, pred_confs, vbs = get_box_logits(imgs,config)
    box,score,pp = predict(imgs,pred_loc, pred_confs, vbs,config.Config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/face_detect/guoshu1/model.ckpt-4403')
        for ip in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/mianhua/open/IMG_20180825_161021.jpg'):
            print(ip)
            img = cv2.imread(ip)
            imgss = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = resize_image(imgss, min_dim=config.image_size, max_dim=config.image_size)

            #img = (org/ 255.0-0.5)*2
            img = org - [123.15, 115.90, 103.06]
            img = np.expand_dims(img, axis=0)
            t = time.time()
            bx,sc,p= sess.run([box,score,pp],feed_dict={imgs:img})
            print(time.time()-t)
            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s]>0.5:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
            if len(bxx) > 0:
                #visual.display_instances(org,np.asarray(bxx)*300)
                finbox = np_utils.revert_image(scale, padding, config.image_size, np.asarray(bxx))
                for ix, s in enumerate(finbox):
                    if cls[ix] == 0:
                        cv2.rectangle(imgss, pt1=(s[0], s[1]), pt2=(s[2], s[3]), color=(0, 255, 0), thickness=10)
                    else:
                        cv2.rectangle(imgss, pt1=(s[0], s[1]), pt2=(s[2], s[3]), color=(255, 255, 0), thickness=10)

                cv2.putText(imgss, 'unopen:open='+str(sum(cls))+':'+str(len(cls)-sum(cls)) , (200,150),cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 0, 0), 12)
                imgss = cv2.cvtColor(imgss,cv2.COLOR_RGB2BGR)
                cv2.imwrite('tt.jpg',imgss)
                visual.display_instances_title(org,np.asarray(bxx)*config.image_size,class_ids=np.asarray(cls),
                                               class_names=['open','unopen'],scores=scores)


train()