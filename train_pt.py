import config
from losses.pt_focul_loss import FocalLoss
from dsl_data.utils import resize_image,resize_image_fixed_size
from tensorflow.contrib import slim
from utils import np_utils
import glob
import cv2
import numpy as np
import time
from data_set import data_gen
import json
from torch import optim
import torch
from models.pt_retinanet import RetinaNet
def train():
    net = RetinaNet(num_classes=11, num_anchors=18)
    net.load_state_dict(torch.load('/home/dsl/all_check/resnet50-19c8e357.pth'),strict=False)

    train_loss = 0

    net.cuda()

    criterion = FocalLoss(num_classes=11)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    gen_bdd = data_gen.get_batch(batch_size=config.batch_size,class_name='lvcai',image_size=config.image_size,max_detect=100)
    for step in range(1000000):

        images, true_box, true_label = next(gen_bdd)
        try:
            loct, conft = np_utils.get_loc_conf_new(true_box, true_label, batch_size=config.batch_size,cfg=config.Config)
        except:
            continue
        images = np.transpose(images, [0,3,1,2])
        images = torch.from_numpy(images)
        true_label = torch.from_numpy(conft).long()
        true_box = torch.from_numpy(loct).float()
        data, labels, boxes = torch.autograd.Variable(images.cuda()), torch.autograd.Variable(true_label.cuda()), torch.autograd.Variable(true_box.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(data)

        loss = criterion(loc_preds, boxes, cls_preds, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

train()