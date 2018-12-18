import numpy as np
import config
from utils import faster_rcnn_utils as utils
class Cfg(object):

    def __init__(self,is_train):
        self.is_train = is_train
        #self.anchors_scals = [128, 256, 512]
        self.image_size = [896, 896]
        self.num_class = 11
        self.batch_size = 8
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.RPN_NMS_THRESHOLD = 0.7
        anchors = config.anchor_gen(config.image_size)
        y_min = anchors[:, 1:2] - anchors[:, 3:] / 2
        y_max = anchors[:, 1:2] + anchors[:, 3:] / 2
        x_min = anchors[:, 0:1] - anchors[:, 2:3] / 2
        x_max = anchors[:, 0:1] + anchors[:, 2:3] / 2
        self.anchors = np.hstack([y_min, x_min, y_max, x_max])

        self.VOC_CLASSES = ('back',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor')

        self.TRAIN_ROIS_PER_IMAGE = 200
        self.DETECTION_MIN_CONFIDENCE = 0.6
        self.DETECTION_MAX_INSTANCES = 100
        self.DETECTION_NMS_THRESHOLD = 0.2
        self.pool_shape = 7
        self.ROI_POSITIVE_RATIO = 0.33
        if is_train:
            self.NMS_ROIS_TRAINING = 2000
        else:
            self.NMS_ROIS_TRAINING = 1000
            self.batch_size = 1

config_instace = Cfg(is_train=True)

