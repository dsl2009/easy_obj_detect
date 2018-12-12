from nms import nms
import json
import numpy as np
import cv2
import os
from dsl_data import visual
from utils import np_utils
from skimage import io
import glob

ll = {'defect0':0.6,'defect1':0.3,'defect2':0.7,'defect3':0.7,'defect4':0.7,'defect5':0.7,
      'defect6':0.7,'defect7':0.6,'defect8':0.7,'defect9':0.3}



def dsl_nms(box, sc, lb_name):
    over_lbs = np_utils.over_laps(box,box)
    zuobiao = np.where(over_lbs>0.1)
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
    lbs[np.where(sc<0.3)]=-1
    return np.where(lbs==0)[0]


def nms_cpu(boxes,score,  overlap):
    if False:
        pick = []
    else:
        trial = np.zeros((len(boxes), 4), dtype=np.float64)
        trial[:] = boxes[:]
        x1 = trial[:, 0]
        y1 = trial[:, 1]
        x2 = trial[:, 2]
        y2 = trial[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # vals = sort(score)
        I = np.argsort(score)
        pick = []
        count = 1
        while (I.size != 0):
            # print "Iteration:",count
            last = I.size
            i = I[last - 1]
            pick.append(i)
            suppress = [last - 1]
            for pos in range(last - 1):
                j = I[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = xx2 - xx1 + 1
                h = yy2 - yy1 + 1
                if (w > 0 and h > 0):
                    o = w * h / area[j]
                    print
                    "Overlap is", o
                    if (o > overlap):
                        suppress.append(pos)
            I = np.delete(I, suppress)
            count = count + 1
    return pick


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
    dt = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/guangdong_round2_test_b_20181106/*.jpg')
    for x in dt:
        d.append(x.split('/')[-1])
    return d
def combin_json():
    file_pths = ['../pred_last.json', '../pred_last_11.json', '../pred_last_13.json']
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
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/guangdong_round2_test_b_20181106/'
    data = json.loads(open('./final.json').read())

    reuslt = data['results']
    print(len(reuslt))
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

show_result()
def combine_new():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/guangdong_round2_test_b_20181106/'
    x = combin_json()
    total_bxx = []
    for rect in x:
        boxes = []
        scores = []
        labels = []
        for b in rect['rects']:

            boxes.append([b['xmin'],b['ymin'],b['xmax'],b['ymax']])
            scores.append(b['confidence'])
            labels.append(b['label'])
        boxes, scores = np.asarray(boxes), np.asarray(scores)
        if boxes.shape[0]>0:
            keep = nms_cpu(boxes, scores, 0.1)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = np.asarray(labels)[keep]

        rectts = []
        for  b in range(boxes.shape[0]):
            dd = {
                'xmin':int(boxes[b][0]),
                'xmax':int(boxes[b][2]),
                'ymin':int(boxes[b][1]),
                'ymax':int(boxes[b][3]),
                'confidence':float(scores[b]),
                'label': labels[b]
                }
            rectts.append(dd)

        total_bxx.append({
                'filename':rect['filename'],
                'rects':rectts
        })
    with open('final.json','w') as f:
        data = {'results':total_bxx}
        f.write(json.dumps(data))
        f.flush()


