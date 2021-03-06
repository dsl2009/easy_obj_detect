import numpy as np
from utils.anchors_gen import gen_ssd_anchors, gen_ssd_anchors_lvcai, gen_ssd_anchors_new,gen_anchors_light_head
remove_norm = {
    'num_classes': 2,
    'feature_maps': [64, 32, 16,8,4],
    'steps': [8, 16, 32,64,128],
    'min_sizes': [10 , 50 ,165 ],
    'max_sizes': [50, 165 ,446 ],
    'aspect_ratios': [[2], [2], [2]],
    'aspect_num':[9,9,9,9,9],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
FACE_CLASSES = ['face']
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
BDD = ['bike', 'bus', 'car', 'motor', 'person', 'rider','traffic light', 'traffic sign', 'train', 'truck']
Lvcai = ['defect0','defect1','defect2','defect3','defect4','defect5','defect6','defect7','defect8','defect9']

rcnn_nms_the = 0.6

Tree = ['tree']
MAX_GT = 100
batch_size = 8
image_size = [256, 256]
mask_pool_shape = [28, 28]
crop_pool_shape = [14, 14]
norm_value = 2.0
mask_weight_loss = 2.0
mask_train = 50
flag = 1
total_fpn = -1

local_voc_dir = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/VOCdevkit/VOCdevkit'
server_voc_dir = '/data_set/data/VOCdevkit'

local_coco_dir = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/train2014'
local_coco_ann = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/coco/raw-data/annotations/instances_train2014.json'

server_coco_dir = '/data_set/data/train2014'
server_coco_ann = '/data_set/data/annotations/instances_train2014.json'

local_check = '/home/dsl/all_check/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt'
server_check = '/data_set/check/inception_v2.ckpt'

local_save = '/home/dsl/all_check/obj_detect/guo_focal_new_coor_aug3'
server_save = '/data_set/check/voc_ssd_yolo'

is_use_group_norm = False
is_use_last = True
if not is_use_last:
    feature_stride = [8, 16, 32,64,128]
    aspect_num = [9,9,9,9,9]
    anchors_size = [16, 32, 64, 128, 256]
else:
    feature_stride = [8, 16, 32, 64]
    #feature_stride = [8, 16, 32, 64]
    #aspect_num = [18, 18, 18, 18, 18]
    #aspect_num = [24, 24, 24, 24, 24]
    #anchors_size = [16, 32, 64, 128]
    #anchors_size = [[14, 18, 24],[30, 36, 42],[48, 54, 64]]
    anchors_size = [[16, 22, 28], [32, 42, 54], [64, 84, 104], [128, 156, 192]]
    aspect_num = [9, 9, 9,9]

anchor_gen = gen_ssd_anchors_new

total_anchor_num = sum([(image_size[0]/x)*(image_size[1]/x)*y for x,y in zip(feature_stride,aspect_num)])
#total_anchor_num = 3072
if flag == 1:
    save_dir = local_save
    check_dir = local_check
    voc_dir = local_voc_dir
    coco_image_dir = local_coco_dir
    annotations = local_coco_ann
elif flag ==2:
    save_dir = server_save
    check_dir = server_check
    voc_dir = server_voc_dir
    coco_image_dir = server_coco_dir
    annotations = server_coco_ann
    batch_size = 32

Config = remove_norm



