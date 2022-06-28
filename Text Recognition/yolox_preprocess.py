import json
import os
import glob
import cv2
import shutil
import re
from pascal_voc_writer import Writer


def merge_train_val(train_dir, val_dir, dst="../train_all/"):
    if not os.path.exists(dst):
        os.mkdir(dst)
    train_image_paths = sorted(glob.glob(train_dir + "*.png"))
    validate_image_paths = sorted(glob.glob(val_dir + "*.png"))
    for path in train_image_paths:
        dst_path = dst + re.split("/|\\\\", path)[-1]
        shutil.copyfile(path, dst_path)
    length = len(train_image_paths)
    del train_image_paths
    for path in validate_image_paths:
        dst_path_index = str(int(re.split("/|\\\\", path)[-1].rstrip(".png")) + length).zfill(6) + ".png"
        dst_path = dst + dst_path_index
        shutil.copyfile(path, dst_path)
    del validate_image_paths
    return


def transform_to_voc(image_dir="../train_all/",
                     original_train_json_path="../mchar_train.json",
                     original_val_json_path="../mchar_val.json",
                     save_dir="../train_all/voc_annotations/"):
    assert os.path.exists(image_dir)
    image_paths = sorted(glob.glob(image_dir + "*.png"))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(original_train_json_path, "r") as f:
        groundtruth = json.load(f)
    f.close()
    start = 100000
    end = 0
    for filename_ in groundtruth:
        start = min(start, int(filename_.rstrip(".png")))
        end = max(end, int(filename_.rstrip(".png")))
        # parse ground truth
        heights = groundtruth[filename_]["height"]
        widths = groundtruth[filename_]["width"]
        tops = groundtruth[filename_]["top"]
        lefts = groundtruth[filename_]["left"]
        labels = groundtruth[filename_]["label"]
        length = len(labels)
        # create the file structure
        cur_img = image_paths[int(filename_.rstrip(".png"))]
        image_width, image_height, image_depth = cv2.imread(cur_img).shape
        writer = Writer(image_dir + filename_, image_width, image_height, image_depth, database="aliyun")
        for i in range(length):
            writer.addObject(str(labels[i]), lefts[i], tops[i], lefts[i]+widths[i], tops[i]+heights[i])
        writer.save(save_dir + filename_.rstrip(".png") + ".xml")
    original_training = end - start + 1
    del groundtruth
    with open(original_val_json_path, "r") as f:
        groundtruth = json.load(f)
    f.close()
    for filename_ in groundtruth:
        heights = groundtruth[filename_]["height"]
        widths = groundtruth[filename_]["width"]
        tops = groundtruth[filename_]["top"]
        lefts = groundtruth[filename_]["left"]
        labels = groundtruth[filename_]["label"]
        length = len(labels)
        filename_ = str(int(filename_.rstrip(".png")) + original_training).zfill(6) + ".png"
        cur_img = image_paths[int(filename_.rstrip(".png"))]
        image_width, image_height, image_depth = cv2.imread(cur_img).shape
        writer = Writer(image_dir + filename_, image_width, image_height, image_depth, database="aliyun")
        for i in range(length):
            writer.addObject(str(labels[i]), lefts[i], tops[i], lefts[i]+widths[i], tops[i]+heights[i])
        writer.save(save_dir + filename_.rstrip(".png") + ".xml")
    return


if __name__ == "__main__":
    # merge_train_val("../mchar_train/", "../mchar_val/")
    transform_to_voc()
