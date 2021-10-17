import pandas as pd
import numpy as np
import random

csv1 = pd.read_csv("detection_label_test_1017_nomosaic.csv", delimiter=",")
csv2 = pd.read_csv("detection_label_test_1017.csv", delimiter=",")
csv3 = pd.read_csv("crnn_submit.csv", delimiter=",")

category = csv1.file_name
det_yolo = csv1.file_code
det_yolo_w = csv2.file_code
resnet = csv3.file_code

res1 = np.array(det_yolo).astype(str)
res2 = np.array(det_yolo_w).astype(str)
res3 = np.array(resnet).astype(str)

category = np.array(category).reshape((40000, ))


def voting(arr1, arr2, arr3):
    """

    :param arr1: yolov4, without mosaic and label smooth
    :param arr2: yolov4 with mosaic and label smooth
    :param arr3: resnet detection
    :return:
    """
    res = arr1[:].astype(str)
    for i in range(40000):
        cur1, cur2, cur3 = arr1[i].strip(".0"), arr2[i].strip(".0"), arr3[i].strip(".0")

        if cur1 != cur2:
            if cur1 != "nan" and cur2 != "nan":
                if len(cur1) > len(cur2):
                    pass
                elif len(cur1) < len(cur2):
                    res[i] = cur2
                else:
                    candidate = [cur1, cur2]
                    res[i] = candidate[random.randint(0, 1)]
            elif cur1 != "nan":
                pass
            elif cur2 != 'nan':
                res[i] = cur2
            else:
                res[i] = cur3
        if len(cur3) > len(res[i]):
            if cur3[len(cur3)-len(res[i]):] == res[i] and cur3[len(cur3)-len(res[i])-1:-1] != res[i]:
                res[i] = cur3[:len(cur3)-len(res[i])] + res[i]
        elif set(cur3).intersection(set(res[i])) is None:
            res[i] = cur3 + res[i]
    return res


arr2 = voting(res1, res2, res3)

final = pd.read_csv("final.csv", delimiter=",")
final_code = final.file_code
final_array = np.array(final_code).astype(str)
cnt = 0
with open("temp.txt", "w") as t:
    for i in range(len(arr2)):

        if arr2[i].strip('.0') != final_array[i].strip('.0'):
            cnt += 1
            vals = str(i) + "\t" + arr2[i].strip('.0') + "\t" + final_array[i].strip('.0') + "\n"
            t.write(vals)
print(cnt)
result = np.vstack((category, arr2)).T
result = pd.DataFrame(result, columns=('file_name', 'file_code'))

result.to_csv("final_1017.csv", index=False)
