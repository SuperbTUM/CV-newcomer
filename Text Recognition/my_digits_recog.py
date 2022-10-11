# ********************************************************** #
import json
from torch.utils.data import DataLoader
import cv2
from torch.utils.data import Dataset
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from dropblock import DropBlock2D
import numpy as np
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
import torchvision.transforms as transforms
import gc
import random
from the_network import resnet50
from prefetch_generator import BackgroundGenerator
import torchvision.models as models

epoch_count = 0
acc_best = 0.
test_init = None
test_epoch = 1
output_dir = './'
rounds = 0
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class ColorJitter(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.uniform(0., 1.) < self.p:
            return sample
        sample["img"] = transforms.ColorJitter(0.3, 0.3, 0.3, 0.2)(sample["img"])
        return sample


class Sharpen(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        # if random.uniform(0., 1.) < self.p:
        #     return sample
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        # sample['img'] = cv2.filter2D(sample['img'], -1, kernel=kernel)
        sample['img'] = transforms.RandomAdjustSharpness(2, self.p)(sample['img'])
        return sample


class Rotation(object):
    def __init__(self, angle=5, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, sample):
        if random.uniform(0.0, 1.0) < self.p:
            return sample
        # ang_rot = np.random.uniform(self.angle) - self.angle / 2
        # h, w, _ = sample["img"].shape
        # transform = cv2.getRotationMatrix2D((w / 2, h / 2), ang_rot, 1)
        # # borderValue = np.mean(sample["img"][0], axis=0).astype(np.float32)
        # sample["img"] = cv2.warpAffine(sample["img"], transform, (w, h),
        #                                borderValue=0)
        sample["img"] = transforms.RandomRotation(self.angle)(sample["img"])
        return sample


class Translation(object):
    def __init__(self, p=0.5):
        self.p = p
        # self.CONSTANT = 1e-3

    def __call__(self, sample):

        if random.uniform(0.0, 1.0) <= self.p:
            return sample
        # h, w, _ = sample["img"].shape
        # trans_range = (w / 10, h / 10)
        # tr_x = trans_range[0] * random.uniform(0.0, 1.0) - trans_range[0] / 2 + self.CONSTANT
        # tr_y = trans_range[1] * random.uniform(0.0, 1.0) - trans_range[1] / 2 + self.CONSTANT
        # transform = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        # # borderValue = np.mean(sample["img"][0], axis=0).astype(np.float32)
        # sample["img"] = cv2.warpAffine(sample["img"], transform, (w, h),
        #                                borderValue=0)
        sample["img"] = transforms.RandomAffine(0, (0.1, 0.1))(sample["img"])
        return sample


# sample: tuple (x:left, y:top, h:height, w:width)
class TextDatasetWithBBox(Dataset):
    def __init__(self, data_json, data_path, transform=None, isTrain=True, seq_len=5):
        super().__init__()
        self.data_json = data_json
        self.data_path = data_path
        self.isTrain = isTrain
        self.seq_len = seq_len

        self.data_label = list()
        for x in self.data_json.keys():
            data = self.data_json[x]
            self.data_label.append(data['label'])
        self.transform = transform
        self.mode = None

    def __len__(self):
        return len(self.data_label)

    def set_mode(self, mode=None):
        self.mode = mode

    def __getitem__(self, idx):
        if self.isTrain:
            gt_label = self.data_label[idx]  # [1,9]
            diff = self.seq_len - len(gt_label)
            gt_label = gt_label[:self.seq_len] + [10 for _ in range(diff)]
        else:
            gt_label = None
        img = cv2.imread(self.data_path[idx]).astype(np.int8)  # select idx-th image
        # img = cv2.resize(img, (224, 128), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1)
        sample = {'img': img, 'label': gt_label}
        if self.transform:
            sample = self.transform(sample)
        sample["img"] = transforms.Resize((224, 128))(sample["img"])
        return sample


# def pred_2_number(preds, cuda=True):
#     preds = nn.Softmax(dim=1)(preds)
#     res = list()
#     for i in range(len(preds)):
#         if cuda:
#             res.append(preds[i].argmax().cpu())
#         else:
#             res.append(preds[i].argmax())
#     return torch.Tensor(res).int()


# class Mish(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return x * torch.tanh(F.softplus(x))
#
#
# mata-ACON
class meta_ACON(nn.Module):
    def __init__(self, channel=64, mode=None):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, channel, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, channel, 1, 1))
        self.mode = mode

    def _cal_beta(self, input):  # (BS, C, H, W)
        if self.mode == 'pixel_wise':
            beta = nn.Sigmoid()(input)
        elif self.mode == 'channel_wise':
            beta = nn.Sigmoid()(torch.sum(input, dim=0))
        elif self.mode == 'layer_wise':
            beta = nn.Sigmoid()(torch.sum(input, dim=(0, 1)))
        elif self.mode is None:
            beta = 1.
        else:
            return NotImplementedError('Invalid mode.')
        return beta

    def forward(self, input):
        output = (self.p1 - self.p2) * input * nn.Sigmoid()(
            self._cal_beta(input) * (self.p1 - self.p2) * input) + self.p2 * input
        return output


class Resnet50Mod(nn.Module):
    def __init__(self, num_class=11):
        super(Resnet50Mod, self).__init__()
        origin_net = models.resnet50()
        self.cnn = nn.Sequential(*list(origin_net.children())[:-1])
        self.hidden_layer = nn.Linear(2048, 128)
        self.dropout = DropBlock2D(block_size=3, drop_prob=0.2)
        self.output1 = nn.Linear(128, num_class)
        self.output2 = nn.Linear(128, num_class)
        self.output3 = nn.Linear(128, num_class)
        self.output4 = nn.Linear(128, num_class)
        self.output5 = nn.Linear(128, num_class)

    def forward(self, img):  # img: bs, c, h, w
        img = self.cnn(img).view(img.size(0), -1)
        assert img.size(1) == 2048
        img = self.hidden_layer(img)
        img = self.dropout(img.view(-1, img.size(1), 1, 1))
        output1 = self.output1(img.view(img.shape[0], -1))
        output2 = self.output2(img.view(img.shape[0], -1))
        output3 = self.output3(img.view(img.shape[0], -1))
        output4 = self.output4(img.view(img.shape[0], -1))
        output5 = self.output5(img.view(img.shape[0], -1))
        return output1, output2, output3, output4, output5


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing. """

    def __init__(self, smoothing=0.02):
        """ Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        target = target.long()
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# step learning rate
# class StepLR(object):
#     def __init__(self, optimizer, step_size=1000, max_iter=10000):
#         self.optimizer = optimizer
#         self.max_iter = max_iter
#         self.step_size = step_size
#         self.last_iter = -1
#         self.base_lrs = list(map(lambda group: group['lr'], optimizer.param_groups))
#
#     def get_lr(self):
#         return self.optimizer.param_groups[0]['lr']
#
#     def step(self, last_iter=None):
#         if last_iter is not None:
#             self.last_iter = last_iter
#         if self.last_iter + 1 == self.max_iter:
#             self.last_iter = -1
#         self.last_iter = (self.last_iter + 1) % self.max_iter
#         for ids, param_group in enumerate(self.optimizer.param_groups):
#             param_group['lr'] = self.base_lrs[ids] * 0.8 ** (self.last_iter // self.step_size)

def pred_2_number(tuple_out):
    res = []
    for out in tuple_out:
        res.append(out.argmax(dim=1))
    return res


def load_network(base_lr=0.01, pretrained=None, cuda=True):
    network = Resnet50Mod()
    if cuda:
        network = network.cuda()
    if pretrained:
        states = torch.load(pretrained)
        network.load_state_dict(states)
        network.eval()
    # optimizer = optim.Adam(network.parameters(), lr=base_lr, weight_decay=0.0001)
    # lr_scheduler = StepLR(optimizer, step_size=step_size, max_iter=max_iter)

    optimizer = optim.SGD(network.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(my_optimizer, len(train_dataset)//batch_size)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, 1e-3)
    loss_function = LabelSmoothing(0.2)
    return network, optimizer, lr_scheduler, loss_function


def test(network, dataset, cuda=True):
    count = 0
    val_dl = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=text_collate, num_workers=num_workers)
    iterator = tqdm(val_dl)
    correct = 0
    for sample in iterator:
        with torch.no_grad():
            imgs = sample["imgs"]  # (bs, num_img, c, h, w)
            img = Variable(imgs)
            if cuda:
                img = img.cuda()
                sample["labels"] = sample['labels'].cuda()
            tuple_out = network(img)
            pred = pred_2_number(tuple_out)
            temp = torch.stack([pred[0] == sample["labels"][:, 0],
                                pred[1] == sample["labels"][:, 1],
                                pred[2] == sample["labels"][:, 2],
                                pred[3] == sample["labels"][:, 3],
                                pred[4] == sample["labels"][:, 4]
                                ], dim=1)
            correct += torch.all(temp, dim=1).sum().item()
            count += batch_size
    status = "acc: {0:.4f}".format(correct / count)
    iterator.set_description(status)

    return correct / count


# ***************************************************************** #

def text_collate(batch):
    imgs = list()
    labels = list()
    for sample in batch:
        imgs.append(sample["img"].float())
        labels.append(sample['label'])

    imgs = torch.stack(imgs)
    labels = torch.Tensor(labels).int()
    batch = {"imgs": imgs, "labels": labels}
    return batch


def WithCuda():
    gpu_num = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if gpu_num > 0 else ''
    return gpu_num > 0


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    train_json = json.load(open('mchar_train.json'))
    train_path = ['mchar_train/' + x for x in train_json.keys()]

    val_json = json.load(open('mchar_val.json'))
    val_path = ['mchar_val/' + x for x in val_json.keys()]

    print("#********************************************# Loading Raw Data Completed!")

    origin_net = resnet50(pretrained=True)
    # origin_net.relu = meta_ACON(mode='layer_wise')
    # origin_net.avgpool = nn.AdaptiveAvgPool2d(1)

    transform = transforms.Compose(
        [
            Sharpen(),
            Rotation(),
            ColorJitter(),
            Translation()
        ]
    )
    train_dataset = TextDatasetWithBBox(train_json, train_path, transform=transform, isTrain=True)
    val_dataset = TextDatasetWithBBox(val_json, val_path, transform=transform, isTrain=True)

    cuda = True if WithCuda() else False
    num_workers = 16
    batch_size = 10

    print("#********************************************# Building Dataset Completed!")

    network, optimizer, lr_scheduler, loss_function = load_network(cuda=cuda)
    while True:
        print('#********************************************# Activating Training Process!')
        if (test_epoch is not None and epoch_count != 0 and epoch_count % test_epoch == 0) or (
                test_init and epoch_count == 0):
            print("Test phase")
            train_dataset.set_mode("test")
            network = network.eval()
            acc = test(network, val_dataset, cuda)
            network = network.train()
            train_dataset.set_mode("train")
            if acc > acc_best:
                rounds = 0
                if output_dir is not None:
                    torch.save(network.state_dict(), os.path.join(output_dir + "resnet50_best.pth"))
                acc_best = acc
            else:
                rounds += 1
                if rounds > 4:
                    print('Test Accuracy does not improve in the past five epochs!\n')
                    print("acc: {:.3f}\tacc_best: {:.3f};".format(acc, acc_best))
                    break
            print("acc: {:.3f}\tacc_best: {:.3f};".format(acc, acc_best))

        loss_mean = list()
        train_dl = DataLoaderX(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                               collate_fn=text_collate)
        iterator = tqdm(train_dl)
        for sample in iterator:
            img_list = sample['imgs']
            if cuda:
                img_list = img_list.cuda()
            optimizer.zero_grad()
            pred = network(img_list)
            labels = sample['labels']
            loss = loss_function(pred[0].cpu(), labels[:, 0]) + \
                   loss_function(pred[1].cpu(), labels[:, 1]) + \
                   loss_function(pred[2].cpu(), labels[:, 2]) + \
                   loss_function(pred[3].cpu(), labels[:, 3]) + \
                   loss_function(pred[4].cpu(), labels[:, 4])
            # loss_function = nn.CrossEntropyLoss()
            # loss = loss_function(pred, Variable(sample['label']).view(-1).long())
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            loss_mean.append(loss.item())
            status = "epoch: {}; lr: {}; loss_mean: {}; loss: {}".format(epoch_count,
                                                                         lr_scheduler.get_last_lr()[0],
                                                                         np.mean(loss_mean), loss.item())
            iterator.set_description(status)
            optimizer.step()
            lr_scheduler.step()
        if epoch_count > 20:
            break
        epoch_count += 1
        if cuda:
            gc.collect()
            torch.cuda.empty_cache()
