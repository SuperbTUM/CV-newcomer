import glob
import torch
from my_digits_recog import DataLoaderX, Resnet50Mod, pred_2_number
import cv2
import numpy as np
import csv


cuda = False


def text_collate_test(batch):
    imgs = list()
    for sample in batch:
        imgs.append(sample['img'].float())

    imgs = torch.stack(imgs)
    batch = {"imgs": imgs}
    return batch


class TextDatasetForTest(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.data_path[idx]).astype(np.float32)  # select idx-th image
        img = cv2.resize(img, (224, 128), interpolation=cv2.INTER_AREA)
        img = torch.from_numpy(img).permute(2, 0, 1)
        sample = {'img': img}
        return sample


if __name__ == "__main__":
    test_path = glob.glob("mchar_test_a/mchar_test_a/*.png")
    test_path.sort()
    test_dataset = TextDatasetForTest(test_path)
    batch_size = 10
    test_dl = DataLoaderX(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False,
                          collate_fn=text_collate_test)
    res = []
    states = torch.load("resnet50_best.pth", map_location=torch.device("cpu"))
    network = Resnet50Mod()
    if cuda:
        network = network.cuda()
    network.load_state_dict(states)
    network.training = False
    for sample in test_dl:
        imgs = sample["imgs"]  # (bs, c, h, w)
        if cuda:
            imgs = imgs.cuda()
        tuple_out = network(imgs)
        preds = pred_2_number(tuple_out)
        preds = torch.Tensor(list(map(lambda x: x.detach().numpy(), preds)))

        for i in range(batch_size):
            candidate = preds[:, i]
            for j in range(len(candidate)):
                if candidate[j] == 10:
                    candidate = candidate[:j]
                    break
            candidate = list(map(int, candidate.tolist()))
            candidate = list(map(str, candidate))
            res.append("".join(candidate))
    with open("detection_labels_resnet.csv", "w", encoding="utf-8", newline="") as g:
        writer = csv.writer(g)
        writer.writerow(["file_name", "file_code"])
        for i in range(len(res)):
            writer.writerow([str(i).zfill(6) + ".png", res[i]])
    g.close()

