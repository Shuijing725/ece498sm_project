import os
import numpy as np
import cv2
from collections import defaultdict

fcn_folder = '/home/shuijing/Desktop/ece498sm_project/runs/1556505662.8736737'
knn_folder = '/home/shuijing/Desktop/ece498sm_project/knn_train'
gt_folder = '/home/shuijing/Desktop/ece498sm_project/data_road/training/gt_image_2'

# gt_folder: the ground truth labels folder (/data_road/training/gt_image_2)
# folder: the folder containing your results
# If you are running it on a single image, just get rid of the outer-most for loop
def calc_iou(gt_folder, folder):
    k = 0
    iou = defaultdict(float)

    print('reading ground truth')
    # read the ground truth
    for file in os.listdir(gt_folder):
        k += 1
        img = cv2.imread(os.path.join(gt_folder, file))
        height, width, _ = img.shape
        # count the tot number of pink pixels
        for i in range(height):
            for j in range(width):
                if img[i, j, 0] == 255 and img[i, j, 1] == 0 and img[i, j, 2] == 255:
                    iou[k] += 1.0

    print('reading predictions')
    k = 0
    pred = defaultdict(float)
    for file in os.listdir(folder):
        k += 1
        img = cv2.imread(os.path.join(folder, file))
        height, width, _ = img.shape
        # count tot number of gray pixels
        for i in range(height):
            for j in range(width):
                if img[i, j, 0] == 128 and img[i, j, 1] == 127 and img[i, j, 2] == 128:
                    pred[k] += 1.0

    print('calculating iou')
    tot = 0.0
    for key in iou:
        tot += pred[key] / iou[key]

    print('IoU = ', tot / len(iou))

calc_iou(gt_folder, knn_folder)