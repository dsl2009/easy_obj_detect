#coding=utf-8
import tensorflow as tf
from losses.ret_nms_loss import get_loss
from dsl_data import visual
import config
from models.dz_model import get_box_logits,predict
from dsl_data.utils import resize_image,resize_image_fixed_size
from tensorflow.contrib import slim
from utils import np_utils
import glob
import cv2
import numpy as np
import time
from data_set import data_gen
import json
from dsl_data import data_loader_multi
import os
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def train():
    img = tf.placeholder(shape=[config.batch_size, config.image_size[0], config.image_size[1], 3], dtype=tf.float32)
    mask = tf.placeholder(shape=[config.batch_size, config.image_size[0], config.image_size[1], 1], dtype=tf.float32)
    loc = tf.placeholder(shape=[config.batch_size, config.total_anchor_num, 4], dtype=tf.float32)
    conf = tf.placeholder(shape=[config.batch_size, config.total_anchor_num], dtype=tf.float32)

    pred_loc, pred_confs, out_put, out_put_mask = get_box_logits(img,config)
    train_tensors = get_loss(conf, loc, pred_loc, pred_confs,config, out_put, mask)
    gen_bdd = data_gen.get_batch_mask(batch_size=config.batch_size,class_name='guoshu',image_size=config.image_size,max_detect=100)
    q = data_loader_multi.get_thread(gen=gen_bdd,thread_num=2)
    global_step = slim.get_or_create_global_step()
    lr = tf.train.exponential_decay(
        learning_rate=0.01,
        global_step=global_step,
        decay_steps=5000,
        decay_rate=0.9,
        staircase=True)

    ig = img+ tf.constant(value=np.asarray([123.15, 115.90, 103.06])/255.0,dtype=tf.float32)
    tf.summary.image('target_mask', mask)
    tf.summary.image('pred_masks', out_put_mask)
    tf.summary.image('ig', ig)
    tf.summary.scalar('lr', lr)
    sum_op = tf.summary.merge_all()
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)

    train_op = slim.learning.create_train_op(train_tensors, optimizer)
    vbs = []
    for s in slim.get_variables():
        if 'resnet_v2_50' in s.name and 'Momentum' not in s.name and 'GroupNorm' not in s.name:
            vbs.append(s)


    def restore(sess):
        saver = tf.train.Saver(vbs)
        saver.restore(sess, config.check_dir)


    sv = tf.train.Supervisor(logdir=config.save_dir, summary_op=None, init_fn=None)

    with sv.managed_session() as sess:
        for step in range(1000000):
            print('       '+' '.join(['*']*(step%10)))
            t = time.time()
            images, true_box, true_label, maskes = q.get()
            try:
                loct, conft = np_utils.get_loc_conf_new(true_box, true_label, batch_size=config.batch_size,cfg=config.Config)
            except:
                continue
            feed_dict = {img: images, loc: loct,
                         conf: conft, mask:maskes}

            ls, step = sess.run([train_op, global_step], feed_dict=feed_dict)
            if step % 10 == 0:
                print('step:' + str(step) +
                      ' ' + 'class_loss:' + str(ls[0]) +
                      ' ' + 'loc_loss:' + str(ls[1])+
                      ' ' + 'mask_loss:' + str(ls[2])+
                      ' ' + 'dice_loss:' + str(ls[3])
                      )
                summaries = sess.run(sum_op, feed_dict=feed_dict)
                sv.summary_computed(sess, summaries)


def detect():
    config.batch_size = 1
    config.image_size = [2048, 2048]
    imgs = tf.placeholder(shape=(1, config.image_size[0], config.image_size[1], 3), dtype=tf.float32,name='input_tensor')
    tf.add_to_collection('input_image',imgs)

    #ig = AddCoords(x_dim=512, y_dim=512)(imgs)
    pred_loc, pred_confs, out_put, out_put_mask = get_box_logits(imgs, config)
    box,score,pp = predict(imgs,pred_loc, pred_confs, config.Config)
    tf.add_to_collection('box', box)
    tf.add_to_collection('score', score)
    tf.add_to_collection('pred', pp)

    saver = tf.train.Saver()
    total_bxx = []

    #builder = tf.saved_model.builder.SavedModelBuilder('server/export')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/dsl/all_check/obj_detect/guo_28/model.ckpt-213644')
        #builder.add_meta_graph_and_variables(sess, ['tag_string'])
        #builder.save()
        images_path = sorted(glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big/*.jpg'))
        for ip in images_path:
            name = ip.split('/')[-1]
            print(name)
            img = cv2.imread(ip)
            imgss = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            org, window, scale, padding, crop = resize_image_fixed_size(imgss,config.image_size)
            img = (org - [123.15, 115.90, 103.06])/255.0
            img = np.expand_dims(img, axis=0)
            t = time.time()
            bx,sc,p, mp= sess.run([box,score,pp,out_put_mask],feed_dict={imgs:img})
            print(time.time()-t)
            bxx = []
            cls = []
            scores = []
            for s in range(len(p)):
                if sc[s]>=0.4:
                    bxx.append(bx[s])
                    cls.append(p[s])
                    scores.append(sc[s])
            rects = []
            if len(bxx) > 0:
                #bxx = np.asarray(bxx)*np.asarray([config.image_size[1],config.image_size[0],config.image_size[1],config.image_size[0]])
                #visual.display_instances_title(org, np.asarray(bxx), class_ids=np.asarray(cls),class_names=config.VOC_CLASSES,scores=scores)
                bxx = np_utils.revert_image(scale, padding, config.image_size, bxx)
                visual.display_instances_title(imgss, bxx, class_ids=np.asarray(cls),class_names=config.Lvcai,scores=scores)
                plt.imshow(mp[0,:,:,0])
                plt.show()
                for ix, b in enumerate(bxx):
                    dd = {
                        'xmin':int(b[0]),
                        'xmax':int(b[2]),
                        'ymin':int(b[1]),
                        'ymax':int(b[3]),
                        'confidence':float(scores[ix]),
                        'label': config.Lvcai[cls[ix]]
                    }
                    rects.append(dd)
            total_bxx.append({
                'filename':name,
                'rects':rects
            })
        with open('pred_last_13.json','w') as f:
            data = {'results':total_bxx}
            f.write(json.dumps(data))
            f.flush()

detect()
