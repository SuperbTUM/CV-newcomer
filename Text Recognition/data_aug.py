# https://blog.csdn.net/qq_32149483/article/details/112056845
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from typing import Tuple, List


class Img_Aug:
    def __init__(self, prob=0.2, crop=True, blur=True, superpixel=True,
                 space_trans=True, sharpen=True, emboss=True,
                 edge_detect=True, noise=True, dropout=True):
        self.prob = prob
        self.crop = crop
        self.blur = blur
        self.superpixel = superpixel
        self.space_trans = space_trans
        self.sharpen = sharpen
        self.emboss = emboss
        self.edge_detect = edge_detect
        self.noise = noise
        self.dropout = dropout

    def __call__(self):
        operations = [iaa.Affine(translate_px={"x": 15, "y": 15},
                                 scale=(0.8, 0.8),
                                 rotate=(-5, 5))]
        if self.crop:
            operations.append(iaa.Crop(percent=(0, self.prob)))
        if self.blur:
            operations.append(iaa.OneOf([iaa.GaussianBlur((0, 3.)),
                                        iaa.AverageBlur(k=(2, 7)),
                                        iaa.MedianBlur(k=(3, 11))]))
        if self.superpixel:
            operations.append(iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0.005, 0.5),
                                                                 n_segments=(16, 28))))
        if self.space_trans:
            operations.append(iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                              iaa.WithChannels(0, iaa.Add((50, 100))),
                                              iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]))
        if self.sharpen:
            operations.append(iaa.Sharpen(alpha=(0., 1.), lightness=(0.75, 1.5)))
        if self.emboss:
            operations.append(iaa.Emboss(alpha=(0., 1.), strength=(0., 2.)))
        if self.edge_detect:
            operations.append(iaa.OneOf([iaa.EdgeDetect(alpha=(0., 0.75)),
                                        iaa.DirectedEdgeDetect(alpha=(0., 0.75), direction=(0, 1))]))
        if self.noise:
            operations.append(iaa.AdditiveGaussianNoise(scale=(0, 128), per_channel=0.5))
        if self.dropout:
            operations.append(iaa.Dropout(p=(0.01, 0.1), per_channel=0.5))
        lenTrans = len(operations)
        seq = iaa.Sequential([iaa.SomeOf(min(5, lenTrans), operations, random_order=True)])
        return seq


class Augmentation:
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, img, bbox=None, label=None, mode="x1y1x2y2") -> Tuple[np.ndarray, dict]:
        # seq_det = self.seq.to_deterministic()
        # bbox should be in x1x2y1y2 format
        if bbox is None:
            return self.seq(image=img)
        if mode == "x1y1x2y2":
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)], shape=img.shape)
        image_aug, bbox_aug = self.seq(image=img, bounding_boxes=bbs)
        location = [bbox_aug[0].x1, bbox_aug[0].y1, bbox_aug[0].x2, bbox_aug[0].y2]
        label = bbox_aug[0].label
        shape = bbox_aug.shape
        bbox_info = {"location": location, "label": label, "shape": shape}
        return image_aug, bbox_info


if __name__ == "__main__":
    seq = Img_Aug()()
    import cv2
    img = cv2.imread("mchar_train/000000.png")
    aug_img, aug_bbox = Augmentation(seq)(img, bbox=[0.5, 0.5, 0.5, 0.5], label=0)
