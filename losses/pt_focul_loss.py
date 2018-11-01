import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def log_sum_exp(self, x):
        """Utility function for computing log_sum_exp while determining
        This will be used to determine unaveraged confidence loss across
        all examples in a batch.
        Args:
            x (Variable(tensor)): conf_preds from conf layers
        """
        x_max = x.data.max()
        return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]

        num_pos = pos.data.long().sum()

        num = loc_preds.size(0)
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        total_cls_loss = torch.zeros( batch_size)
        for b in range(batch_size):
            pred_conf = cls_preds[b]

            conf_t = cls_targets[b]
            p = conf_t==0
            p_true = conf_t>0

            ignore = conf_t ==-1
            conf_t[ignore] =0
            loss_c = (self.log_sum_exp(pred_conf) - pred_conf.gather(1, conf_t.view(-1,1)))
            loss_c[p_true*ignore] = 0
            loss_c = loss_c.view(1,-1)
            _, loss_idx = loss_c.sort(1, descending=True)

            idx = torch.linspace(0, p_true.size()[0]-1, p_true.size()[0])
            num_pos = p_true.long().sum()

            num_neg = num_pos*3

            neg_idx = loss_idx.squeeze(0)[0:num_neg]
            pos_idx = idx[p_true].long().cuda()

            total_idx = torch.cat([neg_idx,pos_idx])

            conf_p = pred_conf[total_idx]
            targets_weighted = conf_t[total_idx]

            loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
            total_cls_loss[b] = loss_c


        N = pos.data.sum().float()
        loss_c = total_cls_loss.sum().cuda()
        loc_loss /= N
        loss_c /= N
        print( loc_loss.cpu().detach().numpy(),loss_c.cpu().detach().numpy())
        return loc_loss+loss_c
