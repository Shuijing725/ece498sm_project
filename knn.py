# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:30:19 2018

@author: Jui Shah
"""

#Road-detection from a traffic video-cam feed


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


img_path = '/home/shuijing/Desktop/ece498sm_project/data_road/testing/image_2'
gt_path = '/home/shuijing/Desktop/ece498sm_project/data_road/testing/gt_image_2'
pred_path = '/home/shuijing/Desktop/ece498sm_project/knn/'

if not os.path.exists(pred_path):
    os.makedirs(pred_path)

l = [('um', 96), ('umm', 94), ('uu', 100)]

for prefix, tot in l:
    for i in range(tot):
        # obtain the full path of each image
        if i < 10:
            path = os.path.join(img_path, str(prefix + '_00000' + str(i) + '.png'))
            cur_gt = os.path.join(gt_path, str(prefix + '_road_00000' + str(i) + '.png'))
        else:
            path = os.path.join(img_path, str(prefix + '_0000' + str(i) + '.png'))
            cur_gt = os.path.join(gt_path, str(prefix + 'road_0000' + str(i) + '.png'))
        image = cv2.imread(path)

        #creating empty image of same size
        height, width, no_use = image.shape
        empty_img = np.zeros((height, width), np.uint8)



        #APPLIED K-MEANS CLUSTERING
        Z = image.reshape((-1,3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 6
        ret,label,center=cv2.kmeans(Z,K,None,criteria,15,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))



        #CONVERTED TO A LUV IMAGE AND MADE EMPTY IMAGE, A MASK
        blur = cv2.GaussianBlur(res2,(15,15),0)
        gray = cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)
        LUV = cv2.cvtColor(blur,cv2.COLOR_RGB2LUV)
        l = LUV[:,:,0]
        v1 = l>80
        v2 = l<150
        value_final = v1 & v2
        empty_img[value_final] = 255
        empty_img[LUV[:,:100,:]] = 0


        #APPLIED BITWISE-AND ON GRAYSCALE IMAGE AND EMPTY IMAGE TO OBTAIN ROAD AND SOME-OTHER IMAGES TOO
        final = cv2.bitwise_and(gray,empty_img)
        final, contours, hierchary = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final = cv2.drawContours(final, contours, -1, 0, 3)



        #FURTHER MASKED THE FINAL IMAGE TO OBTAIN ONLY THE ROAD PARTICLES
        final_masked = np.zeros((height, width), np.uint8)
        v1 = final >=91
        v2 = final <=130
        #v3 = final == 78
        final_masked[v1 & v2] = 255


        #APPLIED EROSION,CONTOURS AND TOP-HAT TO REDUCE NOISE
        kernel = np.ones((3,3),np.uint8)
        final_eroded = cv2.erode(final_masked,kernel,iterations=1)
        final_eroded, contours, hierchary = cv2.findContours(final_eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_masked = cv2.drawContours(final_eroded, contours, -1, 0, 3)

        final_waste = cv2.morphologyEx(final_masked,cv2.MORPH_TOPHAT,kernel, iterations = 2)
        final_waste = cv2.bitwise_not(final_waste)
        final_masked = cv2.bitwise_and(final_waste,final_masked)



        #MADE A LINE ON THE LEFT-BOTTOM OF THE PAGE
        final_masked = cv2.line(final_masked,(40,height),(400,height),255,100)
        #final_masked = cv2.line(final_masked,(width-300,height),(width,height),255,70)



        #USED FLOOD-FILL TO FILL IN THE SMALL BLACK LANES
        final_flood = final_masked.copy()
        h, w = final_masked.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(final_flood,mask,(0,0),255)
        final_flood = cv2.bitwise_not(final_flood)
        final_filled= cv2.bitwise_or(final_masked,final_flood)

        gt = cv2.imread(cur_gt)

        cv2.imwrite(os.path.join(pred_path, str(prefix + '_00000' + str(i) + '.png')), final_filled)

