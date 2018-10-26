from nms import nms
import json
import numpy as np
import cv2
import os
from dsl_data import visual
from utils import np_utils
from skimage import io
import glob

ll = {'defect0':0.6,'defect1':0.4,'defect2':0.7,'defect3':0.7,'defect4':0.7,'defect5':0.7,
      'defect6':0.7,'defect7':0.6,'defect8':0.7,'defect9':0.4}



def dsl_nms(box, sc, lb_name):
    over_lbs = np_utils.over_laps(box,box)
    zuobiao = np.where(over_lbs>0.01)
    lbs = np.zeros(shape=(len(sc),))
    sc = np.asarray(sc)
    ix = np.argmax(sc)
    x = zuobiao[0]
    y = zuobiao[1]
    for i in range(len(x)):
        if x[i] != y[i]:
            if sc[x[i]]>sc[y[i]]:
                lbs[y[i]] = -1

            else:
                lbs[y[i]] = -1
    lbs[ix] = 0
    the = ll[lb_name]
    lbs[np.where(sc<the)]=-1
    return np.where(lbs==0)[0]





def filter_nms(lbs):
    rects = []
    for k in lbs:
        data = np.asarray(lbs[k])
        boxes = data[:,:4]
        scores = data[:,4]

        keep = nms.boxes(boxes, scores, score_threshold=0.3, nms_threshold=0.4)
        keep = dsl_nms(boxes, scores,k)
        newbox = boxes[keep]
        newsc = scores[keep]
        for nb in range(len(newsc)):
            b = newbox[nb]
            cf = newsc[nb]
            dd = {
                'xmin': int(b[0]),
                'xmax': int(b[2]),
                'ymin': int(b[1]),
                'ymax': int(b[3]),
                'confidence': cf,
                'label': k
            }
            rects.append(dd)
    return rects

def get_right():
    d = []
    dt = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/r2testb/*.jpg')
    for x in dt:
        d.append(x.split('/')[-1])
    return d
def combin_json():
    file_pths = ['../pred.json', '../pred_res50.json']
    total_result = []
    for f in file_pths:
        nn = dict()
        data = json.loads(open(f).read())
        reuslt = data['results']
        for r in reuslt:
            nn[r['filename']] = r['rects']
        total_result.append(nn)
    newresults = []
    for x in get_right():
        d = dict()
        d['filename'] = x
        d['rects'] = []
        for k in total_result:
            d['rects'].extend(k[x])
        newresults.append(d)
    return newresults









def combine_result():
    new_result = []
    reuslt = combin_json()
    for x in reuslt:
        labels = dict()
        file_name = x['filename']
        for rect in x['rects']:
            if labels.get(rect['label'],None) is None:
                labels[rect['label']] = [[rect['xmin'],rect['ymin'],rect['xmax'],rect['ymax'],rect['confidence']]]
            else:
                labels[rect['label']].append([rect['xmin'],rect['ymin'],rect['xmax'],rect['ymax'],rect['confidence']])
        new_rects = filter_nms(labels)
        new_result.append({
                'filename':file_name,
                'rects':new_rects
            })

    with open('new_pred.json', 'w') as f:
        d1 = {'results': new_result}
        print(d1)
        f.write(json.dumps(d1))
        f.flush()

def show_result():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/r2testb/'
    data = json.loads(open('./new_pred.json').read())
    reuslt = data['results']
    new_result = []
    for x in reuslt:
        labels = dict()
        file_name = x['filename']
        boxes = []
        scores = []
        lbs = []



        for rect in x['rects']:
            boxes.append([rect['xmin'], rect['ymin'], rect['xmax'], rect['ymax']])
            scores.append(rect['confidence'])
            lbs.append(rect['label'])
        captions = [lb +'_'+ str(s)[0:4] for s, lb in zip(scores, lbs)]
        boxes = np.asarray(boxes)
        print(file_name)
        ig = io.imread(os.path.join(rt, file_name))
        visual.display_instances_title(ig, boxes, class_ids=None, class_names=None, captions=captions)
#combine_result()
show_result()
