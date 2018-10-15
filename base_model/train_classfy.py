from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import json
from base_model import dpn_net
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.metrics import streaming_accuracy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
batch_size = 6
image_size = [512, 704]


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(448, 598)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



def run(trainr,name,cls_num,idx):
    with tf.Graph().as_default() as graph:
        imagenet_data = ImageFolder(trainr, transform=data_transforms['train'])
        data_loader = DataLoader(imagenet_data, batch_size=6, shuffle=True)
        image = tf.placeholder(shape=[batch_size, image_size[0], image_size[1], 3], dtype=tf.float32)
        label = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
        one_hot_label = tf.one_hot(label, depth=cls_num)
        pred = dpn_net.logist(image, num_class=cls_num)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_label, logits=pred)
        total_loss = tf.losses.get_total_loss()
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(
            learning_rate=0.01,
            global_step=global_step,
            decay_steps=10000,
            decay_rate=0.9,
        )
        optm = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        train_op = slim.learning.create_train_op(total_loss, optm)
        predictions = tf.argmax(pred, axis=1)
        probabli = slim.softmax(pred)
        acc, acc_update = streaming_accuracy(predictions, labels=label)
        metric_op = tf.group(acc_update, probabli)
        sv = tf.train.Supervisor(logdir='log')
        with sv.managed_session() as sess:
            for (data, target) in data_loader:
                lb = target.detach().numpy()
                d = data.detach().numpy()
                d = np.transpose(d, axes=[0,2,3,1])
                feed = {image:d, label:lb}
                ls, acc_p,_, g = sess.run([train_op, acc, metric_op, global_step], feed_dict= feed)
                print(ls, acc_p, g)



if __name__ == '__main__':
    train_dr = 'D:/deep_learn_data/luntai/guangdong_round2_train_20181011/guangdong_round2_train_20181011'
    run(train_dr,'luntai',2, 'nn')