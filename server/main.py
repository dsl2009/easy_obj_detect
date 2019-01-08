import time
import tensorflow as tf
import numpy as np
import cv2
import os
from queue import Queue
from threading import Thread
import socket
import json
from server import methods
from matplotlib import pyplot as plt
import glob
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


def detect(qq):
    image_size = [256,256]
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], 'export')
        imgs = sess.graph.get_tensor_by_name('input_tensor:0')
        boxes = sess.graph.get_tensor_by_name('boxes:0')
        score = sess.graph.get_tensor_by_name('score:0')
        pred = sess.graph.get_tensor_by_name('pred:0')
        while True:
            datas = qq.get()
            d = json.loads(datas)
            image_dr = d['task_dr']
            task_id = d['task_id']
            x_min, x_max, y_min, y_max = methods.get_xy(image_dr)
            r_result = []
            w = (x_max - x_min + 1) * 256
            h = (y_max - y_min + 1) * 256
            print(w,h)
            for ixes, pth in enumerate(glob.glob(os.path.join(image_dr, '*.png'))):
                img_bgr = cv2.imread(pth)
                current_loc = pth.split('.')[0].split('/')[-1].split('_')

                c_x, c_y, c_z = int(current_loc[1]), int(current_loc[2]),int(current_loc[0])
                img = img_bgr[:,:,0:3]
                imgss = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                org, window, scale, padding, crop = resize_image_fixed_size(imgss, image_size)
                iges = np.sum(org, axis=2)
                ix = np.where(iges == 0)
                zeros_point = []
                for x in zip(ix[0], ix[1]):
                    zeros_point.append([x[1], x[0]])

                # img = (org/ 255.0-0.5)*2
                img = (org - [123.15, 115.90, 103.06])/255.0
                img = np.expand_dims(img, axis=0)
                t = time.time()
                bx, sc, p = sess.run([boxes, score, pred], feed_dict={imgs: img})
                print(time.time() - t)
                bxx = []
                cls = []
                scores = []
                for s in range(len(p)):
                    if sc[s] > 0.5:
                        x1, y1, x2, y2 = bx[s]
                        center = [int((x1+x2)/2*image_size[0]),int((y1+y2)/2*image_size[1])]
                        if center in zeros_point:
                            continue
                        bxx.append(bx[s])
                        cls.append(p[s])
                        scores.append(sc[s])
                if len(bxx) > 0:
                    finbox = revert_image(scale, padding, image_size, np.asarray(bxx))
                    fin = np.zeros(shape=[finbox.shape[0], 5])
                    r = np.sqrt(((finbox[:,2] - finbox[:, 0]) / 2) ** 2 + ((finbox[:,3] - finbox[:,1]) / 2) ** 2)
                    fin[:,0] = ((finbox[:, 0]+finbox[:,2])/2 -r+(c_x - x_min)*256)/w
                    fin[:,1] = ((finbox[:, 1]+finbox[:,3])/2-r + (c_y- y_min)*256)/h
                    fin[:, 2] = ((finbox[:, 0] + finbox[:,2]) / 2 + r + (c_x - x_min) * 256)/w
                    fin[:, 3] = ((finbox[:, 1] + finbox[:,3]) / 2 + r+(c_y- y_min) * 256)/h
                    fin[:,4] = scores
                    r_result.append(fin)

            r_result = np.vstack(r_result)
            methods.convert_result(r_result, task_id, w, h, sess)







def start():
    q = Queue(maxsize=10)
    num_threads = 1
    for i in range(num_threads):
        worker = Thread(target=detect, args=(q,))
        worker.setDaemon(True)
        worker.start()
    last_task_id = ''
    while True:
        dataes = dataes = {'data': {'task_id': '1ed4ce8a-fb02-499f-9284-90408c961ef8',
                       'task_dr': '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/a733758b-6b57-4923-8a69-42e36cfb5fd0',
                       'scope': {"x_min": 1710742, "z": 21, "y_min": 909283, "x_max": 1710768, "y_max": 909302}},
              'status': 200}
        if dataes['status'] == 200:
            if last_task_id!=dataes['data']['task_id']:
                q.put(json.dumps(dataes['data']))
                last_task_id = dataes['data']['task_id']
        time.sleep(1)




if __name__ == '__main__':
    start()