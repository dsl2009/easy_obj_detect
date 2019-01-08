#coding=utf-8
import config
from utils import np_utils
from data_set import data_gen
import json
from dsl_data import data_loader_multi
import os
from matplotlib import pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def train():

    gen_bdd = data_gen.get_batch_mask(batch_size=config.batch_size,class_name='guoshu',image_size=config.image_size,max_detect=100)
    q = data_loader_multi.get_thread(gen=gen_bdd,thread_num=2)
    d = []
    for step in range(1000000):
        images, true_box, true_label, maskes = q.get()

        loct, conft = np_utils.get_loc_conf_new(true_box, true_label, batch_size=config.batch_size,
                                        cfg=config.Config)


        d.extend(np.reshape(loct[np.where(conft>0)], (1, -1))[0])
        print(np.mean(d))
        print(np.var(d))



train()