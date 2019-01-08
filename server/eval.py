import time
import tensorflow as tf
import numpy as np
import cv2
import os
from queue import Queue
import requests
import socket
import json
from server import methods
from matplotlib import pyplot as plt
from server import handler
import glob
from utils.np_utils import over_laps

def revert_image(scale,padding,image_size,box):

    box = box*np.asarray([image_size[1],image_size[0],image_size[1],image_size[0]])

    box[:, 0] = box[:, 0] - padding[1][0]
    box[:, 1] = box[:, 1] - padding[0][0]
    box[:, 2] = box[:, 2] - padding[1][1]
    box[:, 3] = box[:, 3] - padding[0][1]
    box = box/scale
    box = np.asarray(box,dtype=np.int32)
    return box
def resize_image_fixed_size(image, image_size):
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None
    new_h, new_w = image_size
    scale_h = new_h/h
    scale_w = new_w/w
    scale = min(scale_h, scale_w)
    if scale != 1:
        image = cv2.resize(image, dsize=(round(w * scale), round(h * scale)),interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    top_pad = (new_h - h) // 2
    bottom_pad = new_h - h - top_pad
    left_pad = (new_w - w) // 2
    right_pad = new_w - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image.astype(image_dtype), window, scale, padding, crop


def udp_recon(result):
    udpClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udpClient.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096 * 2)
    #udpClient.settimeout(0.001)
    #msg = json.dumps({'path':img_path}).encode()
    print(result)
    udpClient.sendto(result, ('localhost', 11123))

def get_bound_circle(box,url,ids):
    tt = dict()
    result = dict()
    result['type'] = 'tree'
    result['boundary'] = []
    result['pic_url'] = url
    result['version'] = '0.02'
    for s in range(box.shape[0]):
        x1, y1, x2, y2 = box[s]
        cir = dict()
        dot = {'pix_x':(x1+x2)/2,'pix_y':(y1+y2)/2}
        radius = np.sqrt(((x2-x1)/2)**2+((y2-y1)/2)**2)
        radius = int(radius*10)/10.0
        cir = {'dot':dot,'radius':radius}
        result['boundary'].append(cir)

    tt['data'] = result
    tt['msg_id'] = ids
    udp_recon(json.dumps(tt).encode())


def detect():
    image_size = [2048,2048]
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess_config.gpu_options.allow_growth = True
    he_bing_pth = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big'
    with tf.Session(config=sess_config) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], 'export_cor_aug')
        imgs = sess.graph.get_tensor_by_name('input_tensor:0')
        boxes = sess.graph.get_tensor_by_name('boxes:0')
        score = sess.graph.get_tensor_by_name('score:0')
        pred = sess.graph.get_tensor_by_name('pred:0')
        for image_dr in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/eval_tree/*'):

            print(image_dr)
            handler_dr = os.path.join(he_bing_pth, image_dr.split('/')[-1])
            if os.path.exists(handler_dr):
                for x in glob.glob(os.path.join(handler_dr,'*.jpg')):
                    os.remove(x)
            else:
                os.makedirs(handler_dr)

            handler.hebing_size(handler_dr, 2048, image_dr)


            x_min, x_max, y_min, y_max = methods.get_xy(image_dr)
            r_result = []
            w = (x_max - x_min + 1) * 256
            h = (y_max - y_min + 1) * 256
            print(w,h)
            for ixes, pth in enumerate(glob.glob(os.path.join(handler_dr, '*.jpg'))):
                print(pth)
                img_bgr = cv2.imread(pth)
                current_loc = pth.split('.')[0].split('/')[-1].split('_')

                x_offset, y_offset = int(current_loc[0]), int(current_loc[1])
                img = img_bgr[:,:,0:3]
                imgss = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                org, window, scale, padding, crop = resize_image_fixed_size(imgss, image_size)
                iges = np.sum(org, axis=2)
                ix = np.where(iges == 0)
                zeros_point = []

                # img = (org/ 255.0-0.5)*2
                img = (org - [123.15, 115.90, 103.06])/255.0
                img = np.expand_dims(img, axis=0)
                t = time.time()
                bx, sc, p = sess.run([boxes, score, pred], feed_dict={imgs: img})

                if len(bx) > 0:
                    finbox = revert_image(scale, padding, image_size, bx)

                    finbox[:,1]+=y_offset
                    finbox[:,0]+=x_offset
                    finbox[:, 3] += y_offset
                    finbox[:, 2] += x_offset
                    r_result.append(finbox)

            r_result = np.vstack(r_result)
            sv_dr = os.path.join(handler_dr,'data.npy')
            np.save(sv_dr, r_result)
            with open(os.path.join(handler_dr,'data.json'),'w') as f:
                d = {
                    'w':w,
                    'h':h,
                    'data':sv_dr
                }
                f.write(json.dumps(d))
                f.flush()
            #methods.convert_result(r_result, task_id, w, h, sess)

def get_right():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big'
    url = 'http://agis.xair.cn:8080/aiTrain//external/data.shtml'
    with open('t.txt') as f:
        d = f.readlines()
        for x in d:
            dta = x.replace('\n','').split('\t')
            jiaci = dta[1]
            taskid = dta[0]
            task_dr = os.path.join(rt, taskid)
            print(task_dr)
            if os.path.exists(task_dr):
                boxes = []

                parms = {'act':'putCorrectMessage','type':'tree','taskName':jiaci,'aiVersion':'0.04'}
                r = requests.get(url, params=parms)
                djson = r.json()

                for boundry in djson['data']['boundarys']:
                    if boundry['correction_type'] == 'tree':
                        x_p = []
                        y_p = []
                        for s in boundry['boundary']:
                            x_p.append(int(s['x']))
                            y_p.append(int(s['y']))
                        boxes.append([min(x_p),min(y_p),max(x_p),max(y_p)])
                print(len(boxes))
                boxes = np.asarray(boxes)
                sv_dr = os.path.join(task_dr, 'right.npy')
                np.save(sv_dr, boxes)

def hebing():
    sv = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big'
    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/eval_tree/*'):
        tsk = x.split('/')[-1]
        ig_name = os.path.join(sv,tsk, 'img.jpg')
        methods.hebing_image(x, ig_name)


def draw():
    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big/*'):

        ig_pth = os.path.join(x,'img.jpg')
        pred_img = os.path.join(x,'pred.jpg')
        right_img = os.path.join(x, 'right.jpg')
        pred = os.path.join(x,'data.npy')
        right = os.path.join(x,'right.npy')
        pred = np.load(pred)
        right = np.load(right)
        if x == '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big/180f5da4-b570-4df3-8e1c-db221983039a':
            right = right-256
        print(ig_pth)
        ig = cv2.imread(ig_pth)
        for ix in range(pred.shape[0]):
            box = pred[ix]
            cv2.rectangle(ig, pt1=(box[0], box[1]),pt2=(box[2], box[3]),color=(0, 0,255),thickness=2)
        cv2.imwrite(pred_img, ig)

        ig = cv2.imread(ig_pth)
        for ix in range(right.shape[0]):
            box = right[ix]
            cv2.rectangle(ig, pt1=(box[0], box[1]), pt2=(box[2], box[3]), color=(0, 0, 255), thickness=2)
        cv2.imwrite(right_img, ig)


def clc_acc():
    total = 0
    right_num = 0
    for x in glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big/*'):
        try:
            pred = np.load(os.path.join(x,'data.npy'))
            right = np.load(os.path.join(x,'right.npy'))

            if x == '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/big/180f5da4-b570-4df3-8e1c-db221983039a':
                right = right-256
            ops = over_laps(right, pred)
            d = np.max(ops,axis=0)
            l = len(np.where(d>0.2)[0])
            total+=right.shape[0]
            right_num+=l
            print(x.split('/')[-1], l, right.shape[0], l/right.shape[0])
        except:
            pass
    print('total_acc_map_th=0.5',right_num/total)










def start():
    #detect()
    #get_right()
    #hebing()
    draw()
    clc_acc()




if __name__ == '__main__':
    start()