import numpy as np
import config


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[:,2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:,:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]

    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))  # [A,B]

    union = area_a + area_b - inter
    return inter / union  # [A,B]


def over_laps(boxa,boxb):
    A = boxa.shape[0]
    B = boxb.shape[0]
    b_box = np.expand_dims(boxb,0)
    a = np.repeat(boxa, repeats=B, axis=0)
    b = np.reshape(np.repeat(b_box, repeats=A, axis=0), newshape=(-1, 4))
    d = jaccard_numpy(a, b)
    return np.reshape(d,newshape=(A,B))

def pt_from(boxes):
    xy_min = boxes[:, :2] - boxes[:, 2:] / 2
    xy_max = boxes[:, :2] + boxes[:, 2:] / 2
    return np.hstack([xy_min,xy_max])

def pt_from_nms(boxes):
    y_min = boxes[:, 1:2] - boxes[:, 3:] / 2
    y_max = boxes[:, 1:2] + boxes[:, 3:] / 2
    x_min = boxes[:, 0:1] - boxes[:, 2:3] / 2
    x_max = boxes[:, 0:1] + boxes[:, 2:3] / 2

    return np.hstack([y_min,x_min, y_max, x_max])

def encode(matched, priors, variances):

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]

    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.hstack([g_cxcy, g_wh])  # [num_priors,4]


def get_loc_conf(true_box, true_label,batch_size = 4,cfg =None):
    pri =config.anchor_gen(config.image_size)
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        ix = np.sum(true_box_tm, axis=1)
        true_box_tm = true_box_tm[np.where(ix > 0)]
        labels = labels[np.where(ix > 0)]
        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j
        matches = true_box_tm[best_true_idx]
        conf = labels[best_true_idx] + 1
        conf[best_true < 0.5] = 0
        loc = encode(matches, pri, variances=[0.1, 0.2])
        loc_t[s] = loc
        conf_t[s] = conf
    return loc_t,conf_t

def get_loc_conf_new(true_box, true_label,batch_size = 4,cfg = None):
    pri = config.anchor_gen(config.image_size)
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        ix = (true_box_tm[:, 2]- true_box_tm[:,0])*(true_box_tm[:, 3]- true_box_tm[:,1])
        true_box_tm = true_box_tm[np.where(ix > 1e-6)]
        labels = labels[np.where(ix > 1e-6)]
        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j
            best_true[best_prior_idx[j]] = 1.0
        matches = true_box_tm[best_true_idx]
        conf = labels[best_true_idx] + 1
        conf[best_true <= 0.3] = 0
        b1 = best_true>0.3
        b2 = best_true<=0.5
        conf[b1*b2] = -1
        loc = encode(matches, pri, variances=[0.1, 0.2])
        loc_t[s] = loc
        conf_t[s] = conf
    return loc_t,conf_t


def get_loc_conf_mask_box(true_box, true_mask, true_label,batch_size = 4,cfg = None, mask_shape=[28,28]):
    pri = config.anchor_gen(config.image_size)
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    box_t = np.zeros([batch_size, config.MAX_GT, 4])
    mask_t = np.zeros([batch_size, config.MAX_GT, mask_shape[0], mask_shape[1]])


    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        true_mask_tm = true_mask[s]

        ix = (true_box_tm[:, 2]- true_box_tm[:,0])*(true_box_tm[:, 3]- true_box_tm[:,1])
        true_box_tm = true_box_tm[np.where(ix > 1e-6)]
        labels = labels[np.where(ix > 1e-6)]
        true_mask_tm = true_box_tm[np.where(ix > 1e-6)]

        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j
            best_true[best_prior_idx[j]] = 1.0

        matches = true_box_tm[best_true_idx]
        matches_mask = true_mask_tm[best_true_idx]

        conf = labels[best_true_idx] + 1
        conf[best_true <= 0.3] = 0
        b1 = best_true>0.3
        b2 = best_true<=0.5
        conf[b1*b2] = -1

        loc = encode(matches, pri, variances=[0.1, 0.2])
        loc_t[s] = loc
        conf_t[s] = conf
        box_t[s] = matches
        mask_t[s] = matches_mask

    return loc_t,conf_t, mask_t, box_t



def revert_image(scale,padding,image_size,box):

    box = box*np.asarray([image_size[1],image_size[0],image_size[1],image_size[0]])

    box[:, 0] = box[:, 0] - padding[1][0]
    box[:, 1] = box[:, 1] - padding[0][0]
    box[:, 2] = box[:, 2] - padding[1][1]
    box[:, 3] = box[:, 3] - padding[0][1]
    box = box/scale
    box = np.asarray(box,dtype=np.int32)
    return box

def get_loc_conf_mask(true_box, true_label,batch_size = 4,cfg  = None):

    #pri = get_prio_box(cfg = cfg)
    pri = config.anchor_gen(config.image_size)
    num_priors = pri.shape[0]
    loc_t = np.zeros([batch_size, num_priors, 4])
    conf_t = np.zeros([batch_size, num_priors])
    mask_index = np.zeros([batch_size,num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        ix = np.sum(true_box_tm, axis=1)
        true_box_tm = true_box_tm[np.where(ix > 0)]
        labels = labels[np.where(ix > 0)]
        mask_ix = np.asarray(np.arange(0, labels.shape[0]))
        ops = over_laps(true_box_tm, pt_from(pri))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j

        matches = true_box_tm[best_true_idx]
        mask_t = mask_ix[best_true_idx]+1

        conf = labels[best_true_idx] + 1


        conf[best_true < 0.5] = 0

        mask_t[best_true < 0.5] = 0


        loc = encode(matches, pri, variances=[0.1, 0.2])

        loc_t[s] = loc
        conf_t[s] = conf
        mask_index[s] = mask_t
    return loc_t,conf_t,mask_index



def build_rpn_targets(true_box, true_label,batch_size = 4,cfg = None):
    rpn_nums = 256
    anchors = pri = config.anchor_gen(config.image_size)
    num_priors = anchors.shape[0]
    rpn_box = np.zeros([batch_size, num_priors, 4])
    rpn_labels = np.zeros([batch_size, num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        ix = (true_box_tm[:, 2] - true_box_tm[:, 0]) * (true_box_tm[:, 3] - true_box_tm[:, 1])
        true_box_tm = true_box_tm[np.where(ix > 1e-6)]
        labels = labels[np.where(ix > 1e-6)]
        overlaps = over_laps(true_box_tm, pt_from(anchors))
        anchor_iou_argmax = np.argmax(overlaps, axis=1)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        gt_iou_argmax = np.argmax(overlaps, axis=0)
        ops = over_laps(true_box_tm, pt_from(anchors))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j
            best_true[best_prior_idx[j]] = 1.0

        matches = true_box_tm[best_true_idx]
        conf = labels[best_true_idx] + 1
        conf[best_true>0.5] = 1
        conf[best_true <= 0.3] = 0
        b1 = best_true > 0.3
        b2 = best_true <= 0.5
        conf[b1 * b2] = -1
        cho = np.where(conf==0)[0]
        np.random.shuffle(cho)
        pos_num = len(np.where(conf>0)[0])
        ne_num = pos_num*3
        n_idx = cho[ne_num:]
        conf[n_idx] = -1

        loc = encode(matches, anchors, variances=[0.1, 0.2])

        rpn_box[s] = loc
        rpn_labels[s] = conf
    return rpn_labels, rpn_box



def build_rpn_targets_light_head(true_box, true_label,batch_size = 4,cfg = None):
    rpn_nums = 256
    anchors = pri = config.anchor_gen(config.image_size)
    num_priors = anchors.shape[0]
    rpn_box = np.zeros([batch_size, num_priors, 4])
    rpn_labels = np.zeros([batch_size, num_priors])
    for s in range(batch_size):
        true_box_tm = true_box[s]
        labels = true_label[s]
        ix = (true_box_tm[:, 2] - true_box_tm[:, 0]) * (true_box_tm[:, 3] - true_box_tm[:, 1])
        true_box_tm = true_box_tm[np.where(ix > 1e-6)]
        labels = labels[np.where(ix > 1e-6)]
        overlaps = over_laps(true_box_tm, pt_from(anchors))
        anchor_iou_argmax = np.argmax(overlaps, axis=1)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        gt_iou_argmax = np.argmax(overlaps, axis=0)
        ops = over_laps(true_box_tm, pt_from(anchors))
        best_true = np.max(ops, axis=0)
        best_true_idx = np.argmax(ops, axis=0)
        best_prior = np.max(ops, axis=1)
        best_prior_idx = np.argmax(ops, axis=1)
        for j in range(best_prior_idx.shape[0]):
            best_true_idx[best_prior_idx[j]] = j
            best_true[best_prior_idx[j]] = 1.0

        matches = true_box_tm[best_true_idx]
        conf = labels[best_true_idx] + 1
        conf[best_true>0.5] = 1
        conf[best_true <= 0.3] = 0
        b1 = best_true > 0.3
        b2 = best_true <= 0.5
        conf[b1 * b2] = -1
        cho = np.where(conf==0)[0]
        np.random.shuffle(cho)
        pos_num = len(np.where(conf>0)[0])
        ne_num = pos_num*3
        n_idx = cho[ne_num:]
        conf[n_idx] = -1

        loc = encode(matches, anchors, variances=[0.1, 0.2])

        rpn_box[s] = loc
        rpn_labels[s] = conf
    return rpn_labels, rpn_box




