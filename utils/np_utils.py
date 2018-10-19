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
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride=1):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths,box_heights], axis=2).reshape([-1, 2])

    boxes = np.concatenate([box_centers,box_sizes],axis=1)

    # Convert to corner coordinates (y1, x1, y2, x2)
    #boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                           # box_centers + 0.5 * box_sizes], axis=1)


    return boxes


def gen_multi_anchors(scales, ratios, shape, feature_stride, anchor_stride=1):
    anchors = []
    for s in range(len(feature_stride)):
        an = generate_anchors(scales[s],ratios[s],shape[s],feature_stride[s],anchor_stride=1)
        anchors.append(an)
    return np.vstack(anchors)

def gen_ssd_anchors1():
    #scals = [(36,74,96),(136,198,244),(294,349,420)]
    scals = [(24, 32, 64), (96, 156, 244), (294, 349, 420)]
    ratios = [[0.5,1,2],[0.5,1,2],[0.5,1,2]]
    shape =[(64,64),(32,32),(16,16)]
    feature_stride = [8,16,32]
    anchors = gen_multi_anchors(scales=scals,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/512.0
    out = np.clip(anchors, a_min=0.0, a_max=1.0)
    return out

def gen_ssd_anchors():
    if False:
        size = [16, 32, 64, 128, 256, 512]
        #size = [24, 48, 96, 192, 384, 600]
        feature_stride = [8, 16, 32, 64, 128, 256]
        ratios = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]
    else:
        size = [24, 48, 96, 192, 384]
        feature_stride = [8, 16, 32, 64, 128]
        ratios = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]
    if config.total_fpn!=-1:
        size = size[0:config.total_fpn]
        feature_stride = feature_stride[0:config.total_fpn]
        ratios = ratios[0:config.total_fpn]

    scals = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    sc = [(s * scals[0], s * scals[1], s * scals[2]) for s in size]


    shape = [(config.image_size[0]/x, config.image_size[1]/x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/np.asarray([config.image_size[1],config.image_size[0], config.image_size[1], config.image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out


def gen_ssd_anchors_new():
    size = [16, 32, 64]
    feature_stride = [8, 16, 32, 64]
    ratios = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]
    scals = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    sc = [[s * scals[0], s * scals[1], s * scals[2]] for s in size]

    sc.append([128, 196, 256, 384, 512])
    shape = [(config.image_size[0] / x, config.image_size[1] / x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc, ratios=ratios, shape=shape, feature_stride=feature_stride)
    anchors = anchors / np.asarray(
        [config.image_size[1], config.image_size[0], config.image_size[1], config.image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out

def gen_ssd_anchors_lvcai():
    r = 1.0
    if False:
        size = [16, 32, 64, 128, 256, 512]
        #size = [24, 48, 96, 192, 384, 600]
        feature_stride = [8, 16, 32, 64, 128, 256]
        ratios = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]
    else:
        size = [24*r, 48*r, 96*r, 192*r, 384*r]
        feature_stride = [8, 16, 32, 64, 128]
        ratios = [[0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32]]
    if config.total_fpn!=-1:
        size = size[0:config.total_fpn]
        feature_stride = feature_stride[0:config.total_fpn]
        ratios = ratios[0:config.total_fpn]

    scals = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    sc = [(s * scals[0], s * scals[1], s * scals[2]) for s in size]


    shape = [(config.image_size[0]/x, config.image_size[1]/x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/np.asarray([config.image_size[1],config.image_size[0], config.image_size[1], config.image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out

def gen_anchors_single():

    size = [16, 32, 64, 128, 256, 512]

    feature_stride = [8]
    ratios = [[ 0.5, 1, 2]]

    scals = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    sc = [(16, 32, 64, 128, 256, 512)]
    shape = [(config.image_size[0]/x, config.image_size[1]/x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/np.asarray([config.image_size[1],config.image_size[0], config.image_size[1], config.image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)
    return out

def get_loc_conf(true_box, true_label,batch_size = 4,cfg =None):
    pri = gen_ssd_anchors()
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
    pri = gen_ssd_anchors_lvcai()
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
        b2 = best_true<=0.6
        conf[b1*b2] = -1
        loc = encode(matches, pri, variances=[0.1, 0.2])
        loc_t[s] = loc
        conf_t[s] = conf
    return loc_t,conf_t

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
    pri = gen_ssd_anchors()
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
    anchors = gen_ssd_anchors()
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
        conf[best_true>0.7] = 1
        conf[best_true <= 0.3] = 0
        b1 = best_true > 0.3
        b2 = best_true <= 0.7
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







