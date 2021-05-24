# ********************************************************** #
import json
from torch.utils.data import DataLoader
import cv2
from torch.utils.data import Dataset
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from dropblock import DropBlock2D
import numpy as np
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
import math

epoch_count = 0
acc_best = 0.
test_init = None
test_epoch = 1
output_dir = './'


# sample: tuple (x:left, y:top, h:height, w:width)
class TextDatasetWithBBox(Dataset):
    def __init__(self, data_json, data_path, transform=None, isTrain=True, train_size=8000, val_size=800):
        super().__init__()
        self.data_label = list()
        self.data_bbox = list()
        for x in data_json.keys():
            data = data_json[x]
            self.data_label.append(data['label'])
            single_image_bbox = list()
            for i in range(len(data['label'])):
                single_image_bbox.append((data['left'][i], data['top'][i], data['height'][i], data['width'][i]))
            self.data_bbox.append(single_image_bbox)
        if isTrain:
            self.data_bbox = self.data_bbox[:train_size]
            self.data_label = self.data_label[:train_size]
            self.data_path = data_path[:train_size]
        else:
            self.data_bbox = self.data_bbox[:val_size]
            self.data_label = self.data_label[:val_size]
            self.data_path = data_path[:val_size]
        self.transform = transform

    def __len__(self):
        return len(self.data_label)

    def set_mode(self, mode=None):
        self.mode = mode

    @staticmethod
    def _Resize(bbox):  # (left=x, top=y, height=h, width=w)
        return math.floor(bbox[1]), math.ceil(bbox[1] + bbox[2]), math.floor(bbox[0]), math.ceil(bbox[0] + bbox[3])

    def __getitem__(self, idx):
        gt_label = self.data_label[idx]  # [1,9]
        bbox = self.data_bbox[idx]  # [(loc1), (loc9)] needs recognition
        img = cv2.imread(self.data_path[idx])  # select idx-th image
        img_list = list()
        h, w = 0, 0
        for bb in bbox:
            bb_resize = self._Resize(bb)
            h, w = max(h, bb_resize[1]-bb_resize[0]), max(w, bb_resize[3]-bb_resize[2])
            resized_img = img[bb_resize[0]:bb_resize[1], bb_resize[2]:bb_resize[3]]
            if resized_img.size == 0:
                raise Exception('No image!')
            img_list.append(resized_img)
        # for recognition of each number
        sample = {'img_list': img_list, 'label': gt_label, 'largest_size': (h, w),
                  'is_end': [0] * (len(img_list) - 1) + [1]}
        if self.transform:
            sample = self.transform(sample)
        return sample


def pred_2_number(preds):
    softmax = nn.Softmax(dim=1)
    preds = softmax(preds)
    res = list()
    for i in range(len(preds)):
        res.append(np.argmax(preds[i].detach().numpy()))
    return torch.Tensor(res)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Resnet50Mod(nn.Module):
    def __init__(self, batch_size=20, num_imgs=1):
        super(Resnet50Mod, self).__init__()
        self.batch_size = batch_size
        self.num_imgs = num_imgs
        self.origin_net = models.resnet50(pretrained=True)
        self.origin_net.relu = Mish()
        # the output dims of ave_pool is 1 * 1, according to the paper
        self.origin_net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cnn = nn.Sequential(*list(self.origin_net.children())[:-1])  # without fully connected layer but with
        # ave pooling layer
        # make sure the input dim would be ...*2048
        self.hidden_layer = nn.Linear(2048, 128)
        # self.dropout = nn.Dropout(0.2)
        self.dropout = DropBlock2D(block_size=3, drop_prob=0.2)
        # make sure the output dim would be ...*11
        self.output = nn.Linear(128, 11)  # 11 or 10, 1 * 11
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, img):  # img: bs, c, h, w
        img = self.cnn(img).view(img.size(0), -1)
        # make sure the input is 2 dims
        assert img.size(1) == 2048
        img = self.hidden_layer(img)
        # if use DropBlock2D, then input would be 4 dims
        # img = torch.reshape(img, [bs, img.size(1), 1, 1])
        img = img.view(self.batch_size * self.num_imgs, img.size(1), 1, 1)
        img = self.dropout(img)
        # still make sure the input is 2 dims
        img = img.view(img.size(0), -1)
        output = self.output(img)
        return output
        # preds = self.softmax(output)  # (bs, klass)
        # return self.pred_2_number(preds)

    @staticmethod
    def pred_2_number(preds):
        res = list()
        for i in range(len(preds)):
            res.append(np.argmax(preds[i].detach().numpy()))
        return torch.Tensor(res)

    # def Mish(self, x):
    #     return x * torch.tanh(F.softplus(x))


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):  # x 到底是个啥样的输入：raw output with no probabilities?
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        target = target.long()
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# step learning rate
class StepLR(object):
    def __init__(self, optimizer, step_size=1000, max_iter=10000):
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.step_size = step_size
        self.last_iter = -1
        self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, last_iter=None):
        if last_iter is not None:
            self.last_iter = last_iter
        if self.last_iter + 1 == self.max_iter:
            self.last_iter = -1
        self.last_iter = (self.last_iter + 1) % self.max_iter
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[ids] * 0.8 ** (self.last_iter // self.step_size)


def load_ancillary_functions(network, base_lr=1e-3, step_size=1000, max_iter=10000):
    optimizer = optim.Adam(network.parameters(), lr=base_lr, weight_decay=0.0001)
    lr_scheduler = StepLR(optimizer, step_size=step_size, max_iter=max_iter)
    loss_function = LabelSmoothing(0.2)
    return optimizer, lr_scheduler, loss_function


def test(network, data_loader, cuda=True):
    count = 0
    tp = 0
    iterator = tqdm(data_loader)
    for sample in iterator:
        imgs = sample["img_list"]  # (bs, num_img, c, h, w)
        imgs = imgs.view(imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
        true_label = sample['label'].view(-1)
        segments = sample['is_end'].view(-1)
        img = Variable(imgs)
        if cuda:
            img = img.cuda()
        out = network(img)
        out = pred_2_number(out)
        # 这里需要再做一点处理
        single_out = list()
        single_gt = list()
        for i in segments:
            if segments[i] == 1:  # end
                single_out.append(out[i])
                single_gt.append(true_label[i])
                if single_out == single_out:
                    tp += 1
                count += 1
                single_out.clear()
                single_gt.clear()
            elif segments[i] == 0:
                single_out.append(out[i])
                single_gt.append(true_label[i])
            else:
                continue
        status = "acc: {0:.4f}".format(tp / count)
        iterator.set_description(status)

    return tp / count


# ***************************************************************** #

def text_collate(batch):
    imgs = list()
    labels = list()
    isEnd = list()
    h, w = 0, 0
    seq_len = 0
    # find size to be padded
    for sample in batch:
        h, w = int(max(h, sample['largest_size'][0])), int(max(w, sample['largest_size'][1]))
        seq_len = max(seq_len, len(sample['img_list']))

    color = (255, 255, 255)
    all_black = np.zeros((3, h, w))

    for sample in batch:
        img = list()
        for origin_img in sample['img_list']:
            padding_height = h - origin_img.shape[0]
            top = padding_height >> 1
            bottom = padding_height - top
            padding_width = w - origin_img.shape[1]
            left = padding_width >> 1
            right = padding_width - left
            origin_img = cv2.copyMakeBorder(origin_img.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT,
                                            value=color)
            img.append(origin_img.transpose((2, 0, 1)))
        if len(sample['img_list']) < seq_len:
            remain = seq_len - len(sample['img_list'])
            labels.append(sample['label'] + [10] * remain)
            isEnd.append(sample['is_end'] + [-1] * remain)
            while remain:
                img.append(all_black)
                remain -= 1
        else:
            labels.append(sample['label'])
            isEnd.append(sample['is_end'])
        img = torch.Tensor(img)
        imgs.append(img)

    imgs = torch.stack(imgs)  # each tensor with equal size
    labels = torch.Tensor(labels).int()
    isEnd = torch.Tensor(isEnd)
    batch = {"img_list": imgs, "label": labels, 'is_end': isEnd}
    return batch


if __name__ == '__main__':
    train_json = json.load(open('mchar_train.json'))
    train_path = ['mchar_train/' + x for x in train_json.keys()]

    val_json = json.load(open('mchar_val.json'))
    val_path = ['mchar_val/' + x for x in val_json.keys()]

    print("#********************************************# Loading Raw Data Completed!")

    train_dataset = TextDatasetWithBBox(train_json, train_path, isTrain=True)
    val_dataset = TextDatasetWithBBox(val_json, val_path, isTrain=False)
    # cuda = True if torch.cuda.is_available() else False
    # num_workers = 4 if cuda else 1
    cuda = False
    num_workers = 1
    batch_size = 20

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                          collate_fn=text_collate)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                        collate_fn=text_collate)

    print("#********************************************# Building Dataset Completed!")

    network = None
    while True:
        print('#********************************************# Activating Training Process!')
        if (test_epoch is not None and epoch_count != 0 and epoch_count % test_epoch == 0) or (
                test_init and epoch_count == 0):
            print("Test phase")
            train_dataset.set_mode("test")
            network = network.eval()
            acc = test(network, val_dl, cuda)
            network = network.train()
            train_dataset.set_mode("train")
            if acc > acc_best:
                if output_dir is not None:
                    torch.save(network.state_dict(), os.path.join(output_dir + "_best"))
                acc_best = acc
            print("acc: {}\tacc_best: {};".format(acc, acc_best))

        loss_mean = list()
        iterator = tqdm(train_dl)
        for sample in iterator:  # 输了一个batch进去
            img_list = sample['img_list']
            network = Resnet50Mod(batch_size=batch_size, num_imgs=img_list.shape[1])
            optimizer, lr_scheduler, loss_function = load_ancillary_functions(network)
            img_list = img_list.view(img_list.shape[0] * img_list.shape[1], img_list.shape[2], img_list.shape[3],
                                     img_list.shape[4])
            if cuda:
                network = network.cuda()
                img_list = img_list.cuda()
                sample['label'] = sample['label'].cuda()
            # label_list = sample['label'].view(sample['label'].shape[0] * sample['label'].shape[1], )
            optimizer.zero_grad()
            pred = network(img_list)
            # pred = pred_2_number(pred)
            loss = loss_function(pred, Variable(sample['label']).view(-1))
            # loss_function = nn.CrossEntropyLoss()
            # loss = loss_function(pred, Variable(sample['label']).view(-1).long())
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            loss_mean.append(loss.item())
            status = "epoch: {}; iter: {}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count,
                                                                                   lr_scheduler.last_iter,
                                                                                   lr_scheduler.get_lr(),
                                                                                   np.mean(loss_mean), loss.item())
            iterator.set_description(status)
            optimizer.step()
            lr_scheduler.step()
        if output_dir is not None:
            torch.save(network.state_dict(), os.path.join(output_dir + "_last"))
        if epoch_count > 20:
            break
        epoch_count += 1
