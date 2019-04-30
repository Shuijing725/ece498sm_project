import os
import numpy as np
import cv2
from collections import defaultdict

fcn_folder = '/home/shuijing/Desktop/ece498sm_project/runs/1556505662.8736737'
knn_folder = '/home/shuijing/Desktop/ece498sm_project/knn_train'
gt_folder = '/home/shuijing/Desktop/ece498sm_project/data_road/training/gt_image_2'

def is_gray(pixel):
    if pixel[0] == 128 and pixel[1] == 127 and pixel[2] == 128:
        return True
    else:
        return False

def is_pink(pixel):
    if pixel[0] == 255 and pixel[1] == 0 and pixel[2] == 255:
        return True
    else:
        return False

# gt_folder: the ground truth labels folder (/data_road/training/gt_image_2)
# folder: the folder containing your results
# If you are running it on a single image, just get rid of the outer-most for loop
def calc_iou(gt_folder, folder, method = 'knn'):
    iou = 0.0
    count = 0

    for file in os.listdir(folder):
        # print(file)

        # find the corresponding ground truth in gt_folder
        if method == 'knn':
            if file.startswith('um_'):
                gt_name = 'um_road_' + str(file[-10:])
            elif file.startswith('umm_'):
                gt_name = 'umm_road_' + str(file[-10:])
            else:
                gt_name = 'uu_road_' + str(file[-10:])

        # fcn
        else:
            gt_name = file
        # load predicted img and gt_img
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(gt_folder, gt_name))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        gt = cv2.resize(gt, (width, height))
        intersect = 0.0
        union = 0.0

        # count tot number of gray pixels
        for i in range(height):
            for j in range(width):
                # if intersection
                if is_gray(img[i, j]) and is_pink(gt[i, j]):
                    intersect += 1
                if is_gray(img[i, j]) or is_pink(gt[i, j]):
                    union += 1

        if union != 0:
            count += 1
            iou += intersect / union

    print('IoU = ', iou / count)

calc_iou(gt_folder, fcn_folder, method = 'fcn')