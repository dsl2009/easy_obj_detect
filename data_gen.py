import random
import numpy as np
import config
from dsl_data import aug_utils
from dsl_data import xair_guoshu, mianhua, bdd
def get_batch(batch_size,class_name, is_shuff = True,max_detect = 50,image_size=300):
    if class_name == 'guoshu':
        data_set = xair_guoshu.Tree('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/xair/guoshu/data',
                                    config.image_size)
    elif class_name == 'mianhua':
        data_set = mianhua.MianHua('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/mianhua/open',
                                   config.image_size)
    elif class_name == 'bdd':
        data_set = bdd.BDD(js_file='/home/xair/bdd/labels/labels/bdd100k_labels_images_train.json',
                           image_dr='/home/xair/bdd/bdd100k/images/100k/train',
                                   image_size=config.image_size)
    length = data_set.len()
    idx = list(range(length))
    b = 0
    index = 0
    while True:
        if True:
            if is_shuff and index==0:
                random.shuffle(idx)
                print(idx)
            img, box, lab = data_set.pull_item(idx[index])
            if len(lab) == 0 or len(lab)>100:
                print('no obj')
                index+=1
                continue
            if True:

                if random.randint(0,1)==1:
                   img, box = aug_utils.fliplr_left_right(img,box)
                img = img -[123.15, 115.90, 103.06]

            else:
                img = ((img + [104, 117, 123])/255-0.5)*2.0

            if b== 0:
                images = np.zeros(shape=[batch_size,image_size[0],image_size[1],3],dtype=np.float32)
                boxs = np.zeros(shape=[batch_size,max_detect,4],dtype=np.float32)
                label = np.zeros(shape=[batch_size,max_detect],dtype=np.int32)
                images[b,:,:,:] = img
                boxs[b,:box.shape[0],:] = box
                label[b,:box.shape[0]] = lab
                index=index+1
                b=b+1
            else:
                images[b, :, :, :] = img
                boxs[b, :box.shape[0], :] = box
                label[b, :box.shape[0]] = lab
                index = index + 1
                b = b + 1
            if b>=batch_size:
                yield [images,boxs,label]
                b = 0
            if index>= length:
                index = 0
