import requests
import glob
from skimage import io
import tensorflow as tf
import os
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
def get_task():
    url = 'http://192.168.10.129:9000/app/get_task'
    payload = {'task_type': 'tree'}
    r = requests.get(url=url, params=payload)
    return r.json()

def get_xy(dr):
    x = []
    y = []
    for pth in glob.glob(os.path.join(dr, '*.png')):
        current_loc = pth.split('.')[0].split('/')[-1].split('_')
        c_x, c_y, c_z = int(current_loc[1]), int(current_loc[2]), int(current_loc[0])
        x.append(c_x)
        y.append(c_y)

    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)
    return x_min, x_max, y_min, y_max

def hebing_image(result_dr, sv_image):

    x_min, x_max, y_min, y_max = get_xy(result_dr)
    print(x_min, x_max, y_min, y_max)
    w = (x_max-x_min+1)*256
    h = (y_max-y_min+1)*256
    print(h,w)
    ig = np.zeros(shape=(h, w, 3),dtype=np.uint8)
    print(len(glob.glob(os.path.join(result_dr, '*.png'))))
    print((x_max-x_min)*(y_max-y_min))
    for pth in glob.glob(os.path.join(result_dr, '*.png')):
        img = cv2.imread(pth)
        current_loc = pth.split('.')[0].split('/')[-1].split('_')
        c_x, c_y, c_z = int(current_loc[1]), int(current_loc[2]), int(current_loc[0])

        start_x = c_x -x_min
        start_y = c_y - y_min
        print(start_y*256+256)
        ig[start_y*256:start_y*256+256, start_x*256:start_x*256+256,:] = img[:,:, 0:3]
    cv2.imwrite(sv_image,ig)


def non_max(data, w, h, sess):

    boxes = data[:,0:4]
    score = data[:,4]
    box_tensor = tf.constant(boxes, dtype=tf.float32)
    score_tensor = tf.constant(score, dtype=tf.float32)
    sel = tf.image.non_max_suppression(
        boxes=box_tensor,
        scores=score_tensor,
        iou_threshold=0.3,
        score_threshold=0.5,
        max_output_size=data.shape[0]
    )
    s = sess.run(sel)
    box = boxes[s]*np.asarray([w, h, w, h])
    return box

def convert_circle(data):
    print(data)
    r = (data[:,2] - data[:,0])/2
    x = (data[:,2]+data[:,0])/2
    y = (data[:, 3] + data[:, 1]) / 2
    return x, y, r



def draw_box(data):
    ig = cv2.imread('ss.jpg')
    for x in range(data.shape[0]):
        box = data[x]
        print(box)
        cv2.rectangle(ig, pt1=(box[0], box[1]),pt2=(box[2], box[3]),color=(0, 0,255),thickness=2)
    cv2.imwrite('bo.jpg', ig)

def draw_circle(x, y, r):
    ig = cv2.imread('ss.jpg')
    for x1, y1, r1 in zip(x, y, r):
        cv2.circle(ig, center=(int(x1), int(y1)),radius=int(r1),color=(0, 0,255),thickness=2)
    cv2.imwrite('bo2.jpg', ig)


def tt():
    dt = np.load('guoshu.npy')
    fin = non_max(dt, 9984, 14080)
    x, y, r = convert_circle(fin)
    draw_circle(x,y,r)

def post_result(boundary,task_id):
    url = 'http://192.168.10.129:9000/app/upload_result'
    data = {'ai_version':0.02, 'boundary':boundary, 'task_id':task_id}
    r = requests.post(url, data=data)
    try:
        print(r.text)
    except:
        pass

def update_task():
    tasks = os.listdir('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/biao_zhu/tree/')
    url = 'http://101.37.205.76:8000/app/add_ai_version'
    data = {'type':'tree', 'version':'0.05', 'tasks':json.dumps(tasks)}
    r = requests.get(url, params=data)
    try:
        print(r.text)
    except:
        pass

def filter_box(box):
    mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
    mk = np.where(mj > 0.6)
    box = box[mk]


    mj = (box[:, 3] - box[:, 1]) / (box[:, 2] - box[:, 0])
    mk = np.where(mj < 1.6)
    box = box[mk]
    return box




def convert_result(dt, task_id, w, h, sess):
    #fin = non_max(dt, w, h, sess)
    box = filter_box(dt)
    x, y, r = convert_circle(box)
    print(x,y,r)
    draw_circle(x,y,r)
    boundry = []
    for x1, y1, r1 in zip(x, y, r):
        boundry.append({'dot':{'pix_x':int(x1), 'pix_y':int(y1)},'radius':int(r1)})
    #post_result(json.dumps(boundry), task_id=task_id)
