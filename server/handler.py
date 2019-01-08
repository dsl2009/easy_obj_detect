import requests
import glob
from skimage import io
import tensorflow as tf
import os
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
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

def hebing_image():
    dataes = dataes = {'data': {'task_id': '1ed4ce8a-fb02-499f-9284-90408c961ef8',
                                'task_dr': '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/ca57075f-1846-4d2b-9619-e076f5aaca19',
                                'scope': {"x_min": 1710742, "z": 21, "y_min": 909283, "x_max": 1710768,
                                          "y_max": 909302}},
                       'status': 200}
    d = dataes['data']
    result_dr = d['task_dr']
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
    cv2.imwrite( 'ss.jpg',ig)
    plt.imshow(ig)
    plt.show()


def hebing_size(save_dr, size,result_dr):
    num = int(size/256)
    x_min, x_max, y_min, y_max = get_xy(result_dr)

    w = (x_max - x_min + 1)
    h = (y_max - y_min + 1)

    w_num = int(np.ceil(w / num))
    h_num = int(np.ceil(h / num))
    print(w_num, h_num)
    for w_i in range(w_num):
        for h_j in range(h_num):
            ig = np.zeros(shape=(size, size, 3), dtype=np.uint8)
            ig_name = str(size*w_i)+'_'+str(size*h_j)+'.jpg'
            for ix,x in enumerate(range(x_min+w_i*num, x_min+(w_i+1)*num)):
                for iy, y in enumerate(range(y_min + h_j * num, y_min + (h_j + 1) * num)):
                    pth = glob.glob(os.path.join(result_dr,'*_'+str(x)+'_'+str(y)+'.png'))
                    if len(pth)>0:
                        img = cv2.imread(pth[0])
                        ig[iy * 256:iy * 256 + 256, ix * 256:ix * 256 + 256, :] = img[:, :, 0:3]
            print(os.path.join(save_dr,ig_name))
            cv2.imwrite(os.path.join(save_dr,ig_name),ig)










