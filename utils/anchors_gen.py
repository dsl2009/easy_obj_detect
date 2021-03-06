import numpy as np
import config
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

def gen_ssd_anchors(image_size):
    if False:
        size = [16, 32, 64, 128, 256, 512]
        #size = [24, 48, 96, 192, 384, 600]
        feature_stride = [8, 16, 32, 64, 128, 256]
        ratios = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]
    else:
        size = [24, 48, 96, 192, 384]
        feature_stride = [8, 16, 32, 64, 128]
        ratios = [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]]

    scals = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    sc = [(s * scals[0], s * scals[1], s * scals[2]) for s in size]
    shape = [(image_size[0]/x, image_size[1]/x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/np.asarray([image_size[1],image_size[0], image_size[1], image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out


def gen_ssd_anchors_new(image_size):
    size = config.anchors_size
    feature_stride = config.feature_stride
    ratios = [[0.5,1,2]]*len(feature_stride)
    sc = config.anchors_size
    shape = [(image_size[0] / x, image_size[1] / x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc, ratios=ratios, shape=shape, feature_stride=feature_stride)
    anchors = anchors / np.asarray(
        [image_size[1], image_size[0], image_size[1], image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out

def gen_ssd_anchors_lvcai(image_size):
    r = 1.0
    if False:
        size = [16, 32, 64, 128, 256, 512]
        #size = [24, 48, 96, 192, 384, 600]
        feature_stride = [8, 16, 32, 64, 128, 256]
        ratios = [[0.5, 1, 2]]*len(feature_stride)
    else:
        size = [24*r, 48*r, 96*r, 192*r, 384*r]
        feature_stride = [8, 16, 32, 64, 128]
        ratios = [[0.5, 1, 2, 8, 16, 32], [ 0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32], [0.5, 1, 2, 8, 16, 32]]

    scals = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    sc = [(s * scals[0], s * scals[1], s * scals[2]) for s in size]
    shape = [(image_size[0]/x, image_size[1]/x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc,ratios=ratios,shape=shape,feature_stride=feature_stride)
    anchors = anchors/np.asarray([image_size[1],image_size[0], image_size[1], image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out

def gen_anchors_light_head(image_size):
    size = [24, 48, 96,128]
    feature_stride = [16]
    ratios = [[0.5, 1, 2]]
    sc = [size]
    shape = [(image_size[0] / x, image_size[1] / x) for x in feature_stride]
    anchors = gen_multi_anchors(scales=sc, ratios=ratios, shape=shape, feature_stride=feature_stride)
    anchors = anchors / np.asarray(
        [image_size[1], image_size[0], image_size[1], image_size[0]])
    out = np.clip(anchors, a_min=0.0, a_max=1.0)

    return out
