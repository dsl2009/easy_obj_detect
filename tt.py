from dsl_data import data_loader_multi
import time
from tensorflow.contrib import slim
from nms import nms
import tensorflow as tf
import numpy as np
import cv2

def py_cpu_nms(dets,scores, thresh):
    # dets:(m,5)  thresh:scaler

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []

    index = scores.argsort()[::-1]
    max_rr = index[0]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # because index start from 1
    if len(keep) ==0:
        keep.append(max_rr)
    return keep

num = 1
minxy = np.random.randint(50,145,size=(num,2))
maxxy = np.random.randint(150,200,size=(num,2))
sc = 0.8*np.random.random_sample(num)+0.2
print(sc)
box = np.concatenate((minxy, maxxy),axis=1).astype(np.float32)
t = time.time()
print(nms.boxes(box,sc,score_threshold=0.5, nms_threshold=0.4))
print(py_cpu_nms(box,sc, 0.4))

print(time.time()-t)


