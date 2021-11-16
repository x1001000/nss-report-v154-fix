import re
import cv2
import numpy as np
import pandas as pd
import scipy.signal as signal
import csv
import time
import datetime
import json
import math
import random
import matplotlib.pyplot as plt
# select file
import tkinter as tk
from tkinter import filedialog
import os


# models

from sklearn.cluster import KMeans

# signal
from scipy.signal import find_peaks
from operator import itemgetter, attrgetter

import load_csv as lc
logs = ""

# file type
import os
from os import walk

# logging
import traceback


# create new folder for saving report
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

root = tk.Tk()
root.withdraw()
# create folder
selectpath = lc.patient_test_dir_path
# fall = filedialog.askopenfilename(initialdir=selectpath,title="Select Video File",filetypes=(("Video Files", "*.avi"), ("All Files", "*.*")))
# fpath = os.path.split(fall)
# fpath = fpath[0]
# fpath = filedialog.askdirectory()
# fname = os.path.basename(fall)
# fnameonly = os.path.splitext(fname)

## path processing
os.chdir(selectpath)
createFolder("Report")
target_path = os.path.join(selectpath, "Report")
main_path = target_path
os.chdir(main_path)

file_type = lc.video_type
record = []
for i in range(0, len(lc.patient_test_data)):
    record.append({})


def eyetrack_all(i):
    # Load haar eye detector
    # eye_cascade = cv2.CascadeClassifier("D:\TTCH\Report\haarcascade_eye.xml")
    eye_cascade = cv2.CascadeClassifier("/Users/peterliang/Desktop/Neruobit_Python/Report/haarcascade_eye.xml")
    eye_cascade.load("/Users/peterliang/Desktop/Neruobit_Python/Report/haarcascade_eye.xml")

    # Define the csv and create Nystagmus waveform
    # myFile = open('Pupil_center.csv', 'w')
    target_path_csv_record = os.path.join(target_path, lc.patient_test_data[i]['file_name']) + ".csv"
    with open(target_path_csv_record, 'w', newline='') as myFile:
        myFields = ['Timestamps', 'Righteye_X', 'Righteye_Y', 'Lefteye_X', 'Lefteye_Y', 'Righteye_Area', 'Lefteye_Area',
                    'Righteye_Center_X', 'Righteye_Center_Y', 'Lefteye_Center_X', 'Lefteye_Center_Y',
                    'Righteye_RET_Area', 'Lefteye_RET_Area',
                    'Righteye_Hist_Mean', 'Lefteye_Hist_Mean', "Righteye_Error_Message", "Lefteye_Error_Message"]
        writer = csv.DictWriter(myFile, fieldnames=myFields)
        writer.writeheader()

    # Setting the initial value 0 and (0,0) for pupil center and frame image
    # Avoiding none result error
    currentFrame = 0
    ellipse = ((320.0, 240.0), (0.0, 0.0), 0.0)
    ellipse1 = ((960.0, 240.0), (0.0, 0.0), 0.0)
    center = (320, 240)
    center1 = (960, 240)
    center_x = 0
    center_y = 0
    center_x1 = 0
    center_y1 = 0
    type1 = 0
    type2 = 0
    type3 = 0
    type4 = 0
    type5 = 0
    type6 = 0
    # upload package data
    st_All = []
    center_x_All = []
    center_y_All = []
    center_x1_All = []
    center_y1_All = []

    # user parameter definition
    parameter_crop_gray_hist_mean = 35  # code line 130
    parameter_crop_gray1_hist_mean = 35  # code line 196
    parameter_area_min = 350  # code line 146
    parameter_area1_min = 350  # code line 212
    parameter_area_max = 4500  # code line 146
    parameter_area1_max = 4500  # code line 212

    parameter_area_max_r = 50000
    parameter_area_min_r = 4500

    # righteye_sqr_area = [np.nan for i in range(125)]
    # righteye_x = [np.nan for i in range(125)]
    # righteye_y = [np.nan for i in range(125)]
    # righteye_area_buffer = [np.nan for i in range(125)]
    righteye_area = "N/A"
    righteye_center_x = "N/A"
    righteye_center_y= "N/A"
    righteye_ret_area = "N/A"
    righteye_hist_mean = "N/A"
    last_pos = ellipse
    re_d = 20

    # lefteye_sqr_area = [np.nan for i in range(125)]
    # lefteye_x = [np.nan for i in range(125)]
    # lefteye_y = [np.nan for i in range(125)]
    # lefteye_area_buffer = [np.nan for i in range(125)]
    lefteye_area = "N/A"
    lefteye_center_x = "N/A"
    lefteye_center_y = "N/A"
    lefteye_ret_area = "N/A"
    lefteye_hist_mean = "N/A"
    last_pos1 = ellipse1
    le_d = 20

    if 'fixation' in lc.patient_test_data[i]['file_name']:
        parameter_area_min = 500  # code line 146
        parameter_area1_min = 500  # code line 212
        parameter_area_max = 10000  # code line 146
        parameter_area1_max = 10000  # code line 212
        parameter_area_min_r = 4000
        re_d = 30
        le_d = 30

    # Start to record webcam video forever
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # write the USB stream frame
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            out.write(frame)
            # print(gray.shape)

            ##### Right eye #####
            crop_gray = gray[0:480, 0:640]  # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

            cv2.circle(crop_gray, (320, 240), 1, (255, 0, 0), -1)
            eyes = eye_cascade.detectMultiScale(crop_gray)
            righteye_error = "Normal"

            if len(eyes) == 0:
                # print("No right eye detected in Haarcascade_classifyer: " + str(currentFrame))
                righteye_error = "No right eye detected."

                # crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 0)
                # crop_gray_hist_mean = parameter_crop_gray_hist_mean
                # try:
                #     iter_threshold = int(round(crop_gray_hist_mean*(-1/5)+12))
                # except:
                #     iter_threshold = 3
                #
                # ret,thresh = cv2.threshold(crop_gray,crop_gray_hist_mean,255,cv2.THRESH_BINARY)
                #
                # kernel = np.ones((3, 3), np.uint8)
                # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iter_threshold)
                #
                # cont_img = ~closing.copy()
                # contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                #
                # ellipse_tmp_r = 0
                # tmp_last = 400
                # for cnt in contours:
                #     if len(cnt) < 5:
                #         break
                #
                #     if area < parameter_area_min or area > parameter_area_max:
                #         continue
                #     #ellipse = cv2.fitEllipse(cnt)
                #     ellipse_tmp = cv2.fitEllipse(cnt)
                #     tmp_x = np.array([ellipse_tmp[0][0], 320])
                #     tmp_y = np.array([ellipse_tmp[0][1], 240])
                #     tmp_d = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                #     if (ellipse_tmp[1][0]/ellipse_tmp[1][1] > ellipse_tmp_r) and (tmp_last > tmp_d):
                #        ellipse_tmp_r = ellipse_tmp[1][0]/ellipse_tmp[1][1]
                #        ellipse = cv2.fitEllipse(cnt)
                #        tmp_x = np.array([ellipse[0][0], 320])
                #        tmp_y = np.array([ellipse[0][1], 240])
                #        tmp_last = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                #     else:
                #        ellipse_tmp_r = ellipse_tmp_r
                #        tmp_last = tmp_last
                #
                # cv2.ellipse(crop_gray, ellipse, (255,0,0), 2)
                # center = ellipse[0]

            else:
                # print("Frame: ", currentFrame)
                for (ex, ey, ew, eh) in eyes:

                    righteye_img = crop_gray[(ey + int(eh / 4) - re_d):(ey + int(eh * 3 / 4) + re_d),
                                   (ex + int(ew / 4) - re_d):(ex + int(ew * 3 / 4) + re_d)]
                    # righteye_img = crop_gray[ey:(ey + eh), ex:(ex + ew)]
                    hist = cv2.calcHist([righteye_img], [0], None, [256], [0, 256])
                    # indices = find_peaks(hist.flatten(), height=100, distance=15)
                    indices = find_peaks(hist.flatten(), prominence=25, distance=15)

                    try:
                        crop_gray_hist_mean = int(
                            (hist[indices[0][0]] * indices[0][0] + hist[indices[0][1]] * indices[0][1]) / (
                                    hist[indices[0][0]] + hist[indices[0][1]]))
                        if crop_gray_hist_mean > 60 or crop_gray_hist_mean < 15:
                            crop_gray_hist_mean = parameter_crop_gray_hist_mean
                            # center = last_pos
                            # cv2.ellipse(crop_gray, center, (255, 0, 0), 2)
                            # center = center[0]
                            # cv2.circle(crop_gray, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
                            # righteye_error = "HIST_MEAN ERROR"
                            # # print("HIST_MEAN ERROR")
                            # break

                    except:
                        crop_gray_hist_mean = parameter_crop_gray_hist_mean

                    # plt.plot(hist)
                    # plt.plot(indices[0], hist[indices[0]], 'r.')
                    # plt.show()
                    # plt.clf()
                    # img = cv2.rectangle(crop_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    # print(hist)

                    # print('EX', ex, "|", np.nanmedian(righteye_x) + 2 * np.nanstd(righteye_x))
                    # print('EY', ey, "|", np.nanmedian(righteye_x) - 2 * np.nanstd(righteye_x))
                    # # print('AREA', (ew * eh), "|",
                    #       np.nanmedian(righteye_sqr_area) - 2 * np.nanstd(righteye_sqr_area))

                    # if (ew * eh) / 4 < (np.nanmedian(righteye_sqr_area) - 2 * np.nanstd(
                    #         righteye_sqr_area)) and currentFrame > 25:
                    #     print('RE Too small haardetector: ' + str(currentFrame))
                    #     break
                    if currentFrame > 10:
                        if (ew / 2 + 2 * re_d) * (eh / 2 + 2 * re_d) < parameter_area_min_r:
                            center = last_pos
                            cv2.ellipse(crop_gray, center, (255, 0, 0), 2)
                            center = center[0]
                            cv2.circle(crop_gray, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
                            # print('RE Too small haardetector: ' + str(currentFrame))
                            righteye_error = "RET too small."
                            break
                        elif (ew / 2 + 2 * re_d) * (eh / 2 + 2 * re_d) > parameter_area_max_r:
                            center = last_pos
                            cv2.ellipse(crop_gray, center, (255, 0, 0), 2)
                            center = center[0]
                            cv2.circle(crop_gray, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
                            # print('RE Too big haardetector: ' + str(currentFrame))
                            righteye_error = "RET too big."
                            break
                    # elif ex > (np.nanmedian(righteye_x) + 2 * np.nanstd(righteye_x)) and currentFrame > 125:
                    #     print('RE Too right: ' + str(currentFrame))
                    #     break
                    # elif ey < (np.nanmedian(righteye_y) - 2 * np.nanstd(righteye_y)) and currentFrame > 125:
                    #     print('RE Too low: ' + str(currentFrame))
                    #     break

                    # righteye_sqr_area[currentFrame % 125] = (ew * eh) / 4
                    # righteye_x[currentFrame % 125] = ex + 1/4*ew
                    # righteye_y[currentFrame % 125] = ey + 1/4*eh
                    righteye_ret_area = (ew / 2 + 2 * re_d) * (eh / 2 + 2 * re_d)

                    # cv2.circle(crop_gray, (ex + int(1 / 4 * ew)-re_d, ey + int(1 / 4 * eh)-re_d), 7, (255, 255, 255), 2) # mark left-up point of ret
                    cv2.rectangle(crop_gray, (ex + int(ew / 4) - re_d, ey + int(eh / 4) - re_d),
                                  (ex + int(ew * 3 / 4) + re_d, ey + int(eh * 3 / 4) + re_d), (255, 255, 255),
                                  2)  # mark ret
                    # cv2.rectangle(crop_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    # print(str(np.nanmedian(righteye_sqr_area))+'_'+str(currentFrame))

                    righteye_img = cv2.GaussianBlur(righteye_img, (5, 5), 0)
                    # cv2.imshow('righteye_img', righteye_img)

                    # print(crop_gray_hist_mean)

                    righteye_hist_mean = crop_gray_hist_mean

                    try:
                        iter_threshold = int(round(crop_gray_hist_mean * (-1 / 5) + 12))
                    except:
                        iter_threshold = 3

                    ret, thresh = cv2.threshold(righteye_img, crop_gray_hist_mean, 255, cv2.THRESH_BINARY)
                    # plt.hist(thresh.ravel(),256)

                    kernel = np.ones((2, 2), np.uint8)

                    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iter_threshold)
                    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    cont_img = ~closing.copy()
                    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    cont_img = cv2.cvtColor(cont_img, cv2.COLOR_GRAY2BGR)
                    # print(len(contours))

                    contours = sorted(contours, key=lambda x: len(x), reverse=True)

                    for i in range(len(contours)):
                        if i % 3 == 0:
                            cv2.drawContours(cont_img, contours, i, (255, 0, 0), 3)
                        elif i % 3 == 1:
                            cv2.drawContours(cont_img, contours, i, (0, 255, 0), 3)
                        else:
                            cv2.drawContours(cont_img, contours, i, (0, 0, 255), 3)

                    # ellipse_tmp_r = 0
                    # tmp_last = 400
                    cnt_status = 0
                    for i in range(len(contours)):
                        # print('Pupil Area', area)
                        if len(contours[i]) < 5:
                            break

                        try:
                            ellipse_tmp = cv2.fitEllipse(contours[i])
                            area_tmp = math.pi * ellipse_tmp[1][0] * ellipse_tmp[1][1] / 4

                            # print("Trial AREA: ", area1_tmp)
                            # print("Trial RATE: ", ellipse1_tmp[1][0] / ellipse1_tmp[1][1])

                            if area_tmp < parameter_area_min or area_tmp > parameter_area_max:
                                if i == len(contours) - 1:
                                    cnt_status = 1
                                    righteye_error = "CNT Area ERROR"
                                continue

                            if (ellipse_tmp[1][0] / ellipse_tmp[1][1]) > (1 / 0.5) or (
                                    ellipse_tmp[1][0] / ellipse_tmp[1][1]) < 0.5:
                                if i == len(contours) - 1:
                                    cnt_status = 1
                                    righteye_error = "CNT Rate ERROR"
                                continue

                            ellipse = cv2.fitEllipse(contours[i])
                            area = math.pi * ellipse[1][0] * ellipse[1][1] / 4
                            righteye_area = area

                            # print("Correct AREA: ", area1)
                            # print("Correct RATE: ", ellipse1[1][0] / ellipse1[1][1])

                            break
                        except:
                            break

                        # area = cv2.contourArea(cnt)

                        # print('Pupil Area', area, (
                        # np.nanmedian(righteye_area_buffer) + 2 * np.nanstd(righteye_area_buffer),
                        # (np.nanmedian(righteye_area_buffer) - 2 * np.nanstd(righteye_area_buffer))))
                        # if (area > (np.nanmedian(righteye_area_buffer) + 2 * np.nanstd(
                        #         righteye_area_buffer)) or area < (
                        #             np.nanmedian(righteye_area_buffer) - 2 * np.nanstd(
                        #             righteye_area_buffer))) and currentFrame > 125:
                        #     print("RE area is out of range.")
                        #     break

                        # if area < parameter_area_min or area > parameter_area_max:
                        #     continue
                        # ellipse = cv2.fitEllipse(cnt)

                        # tmp_x = np.array([ellipse_tmp[0][0], 320])
                        # tmp_y = np.array([ellipse_tmp[0][1], 240])
                        # tmp_d = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                        # if (ellipse_tmp[1][0] / ellipse_tmp[1][1] > ellipse_tmp_r) and (tmp_last > tmp_d):
                        #     ellipse_tmp_r = ellipse_tmp[1][0] / ellipse_tmp[1][1]
                        #     ellipse = cv2.fitEllipse(cnt)
                        #     tmp_x = np.array([ellipse[0][0], 320])
                        #     tmp_y = np.array([ellipse[0][1], 240])
                        #     tmp_last = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                        #
                        # else:
                        #     ellipse_tmp_r = ellipse_tmp_r
                        #     tmp_last = tmp_last

                    # cv2.ellipse(righteye_img, ellipse, (255,0,0), 2)

                    try:
                        if cnt_status == 1:
                            center = last_pos
                            cv2.ellipse(crop_gray, center, (255, 0, 0), 2)
                            center = center[0]
                            cv2.circle(crop_gray, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
                            # print('RE CNT ERROR: ' + str(currentFrame))
                            break
                        else:
                            center_tmp = ellipse[0]
                            # center = (center_tmp[0] + ex, center_tmp[1] + ey)
                            center = (
                                (center_tmp[0] + ex + 1 / 4 * ew - re_d, center_tmp[1] + ey + 1 / 4 * eh - re_d),
                                ellipse[1], ellipse[2])
                            cv2.ellipse(crop_gray, center, (255, 0, 0), 2)
                            last_pos = center
                            center = center[0]
                            cv2.circle(crop_gray, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
                            righteye_center_x = int(center[0])
                            righteye_center_y = int(center[1])
                            break
                    except:
                        break

            # print("RIGHTEYE CENTER: ", center[0], center[1])
            center_x = -((center[0] - 320) * 0.05 / 12 * 180 / math.pi)
            center_y = -((center[1] - 240) * 0.05 / 12 * 180 / math.pi)

            # cv2.circle(crop_gray, (int(center[0]),int(center[1])), 5, (255,0,0), -1)
            # cv2.imshow('Neurobit',crop_gray)

            # classic eyeball diameter is 12 mm (r)
            # length fo Jim's eyes are 24+-1 mm (S)
            # S = r*theta (S:length of eyeball surface, theta: angle of eyeball)
            # left eye amd right eye are 480 pixels = 24 mm (1 pixels = 0.05 mm)
            # theta = pixel*0.05/12*180/pi (degree)

            ##### Left eye #####
            crop_gray1 = gray[0:480, 640:1280]  # Crop from x, y, w, h -> 100, 200, 300, 400
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

            cv2.circle(crop_gray1, (320, 240), 1, (255, 0, 0), -1)
            eyes1 = eye_cascade.detectMultiScale(crop_gray1)
            lefteye_error = "Normal"
            if len(eyes1) == 0:
                # print("No left eye detected in Haarcascade_classifyer_" + str(currentFrame))
                lefteye_error = "No left eye detected."

                # crop_gray1 = cv2.GaussianBlur(crop_gray1, (5, 5), 0)
                # crop_gray_hist_mean = crop_gray_hist_mean
                # try:
                #     iter_threshold = int(round(crop_gray_hist_mean/10))
                # except:
                #     iter_threshold = 3
                #
                # ret1,thresh1 = cv2.threshold(crop_gray1,crop_gray_hist_mean,255,cv2.THRESH_BINARY)
                #
                # kernel = np.ones((3, 3), np.uint8)
                # closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=iter_threshold)
                #
                # cont_img1 = ~closing1.copy()
                # contours, hierarchy = cv2.findContours(cont_img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                #
                # ellipse_tmp_r = 0
                # tmp_last = 400
                # for cnt in contours:
                #     if len(cnt) < 5:
                #         break
                #     area = cv2.contourArea(cnt)
                #     if area < parameter_area_min or area > parameter_area_max:
                #         continue
                #     ellipse_tmp = cv2.fitEllipse(cnt)
                #     tmp_x = np.array([ellipse_tmp[0][0], 320])
                #     tmp_y = np.array([ellipse_tmp[0][1], 240])
                #     tmp_d = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                #     if (ellipse_tmp[1][0]/ellipse_tmp[1][1] > ellipse_tmp_r) and (tmp_last > tmp_d):
                #        ellipse_tmp_r = ellipse_tmp[1][0]/ellipse_tmp[1][1]
                #        ellipse = cv2.fitEllipse(cnt)
                #        tmp_x = np.array([ellipse[0][0], 320])
                #        tmp_y = np.array([ellipse[0][1], 240])
                #        tmp_last = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                #     else:
                #        ellipse_tmp_r = ellipse_tmp_r
                #        tmp_last = tmp_last
                #
                # cv2.ellipse(crop_gray1, ellipse, (255,0,0), 2)
                # center1 = ellipse[0]

            else:
                for (ex1, ey1, ew1, eh1) in eyes1:
                    lefteye_img = crop_gray1[(ey1 + int(eh1 / 4) - le_d):(ey1 + int(eh1 * 3 / 4) + le_d),
                                  (ex1 + int(ew1 / 4) - le_d):(ex1 + int(ew1 * 3 / 4) + le_d)]
                    hist = cv2.calcHist([lefteye_img], [0], None, [256], [0, 256])

                    # indices = find_peaks(hist.flatten(), height=100, distance=15)
                    indices = find_peaks(hist.flatten(), prominence=25, distance=15)

                    try:
                        crop_gray1_hist_mean = int(
                            (hist[indices[0][0]] * indices[0][0] + hist[indices[0][1]] * indices[0][1]) / (
                                    hist[indices[0][0]] + hist[indices[0][1]]))
                        if crop_gray1_hist_mean > 60 or crop_gray1_hist_mean < 15:
                            # center1 = last_pos1
                            # cv2.ellipse(crop_gray1, center1, (255, 0, 0), 2)
                            # center1 = center1[0]
                            # cv2.circle(crop_gray1, (int(center1[0]), int(center1[1])), 5, (255, 0, 0), -1)
                            # lefteye_error = "HIST_MEAN ERROR"
                            # # print("HIST_MEAN ERROR")
                            # break
                            crop_gray1_hist_mean = parameter_crop_gray1_hist_mean
                    except:
                        crop_gray1_hist_mean = parameter_crop_gray1_hist_mean

                    # print(crop_gray1_hist_mean)
                    # plt.plot(hist)
                    # plt.plot(indices[0], hist[indices[0]], 'r.')
                    # plt.show()
                    # plt.clf()

                    # img = cv2.rectangle(crop_gray1, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (0, 255, 0), 2)
                    # print(hist)
                    # if currentFrame > 300:

                    # print('ex1', ex1, "|", np.nanmedian(lefteye_x) + 2 * np.nanstd(lefteye_x))
                    # print('ey1', ey1, "|", np.nanmedian(lefteye_x) - 2 * np.nanstd(lefteye_x))
                    # print('AREA', (ew1 / 2 + le_d) * (eh1 / 2 + le_d))

                    if currentFrame > 10:
                        if (ew1 / 2 + 2 * le_d) * (eh1 / 2 + 2 * le_d) < parameter_area_min_r:
                            center1 = last_pos1
                            cv2.ellipse(crop_gray1, center1, (255, 0, 0), 2)
                            center1 = center1[0]
                            cv2.circle(crop_gray1, (int(center1[0]), int(center1[1])), 5, (255, 0, 0), -1)
                            # print('LE Too small haardetector: ' + str(currentFrame))
                            lefteye_error = "RET too small."
                            break
                        elif (ew1 / 2 + 2 * le_d) * (eh1 / 2 + 2 * le_d) > parameter_area_max_r:
                            center1 = last_pos1
                            cv2.ellipse(crop_gray1, center1, (255, 0, 0), 2)
                            center1 = center1[0]
                            cv2.circle(crop_gray1, (int(center1[0]), int(center1[1])), 5, (255, 0, 0), -1)
                            # print('LE Too big haardetector: ' + str(currentFrame))
                            lefteye_error = "RET too big."
                            break

                    # if (ew1 * eh1) / 4 < (
                    #         np.nanmedian(lefteye_sqr_area) - 2 * np.nanstd(lefteye_sqr_area)) and currentFrame > 25:
                    #     print('LE Too small haardetector: ' + str(currentFrame))
                    #     break
                    # elif (ex1 > (np.nanmedian(lefteye_x) + 2 * np.nanstd(lefteye_x)) or ex1 < (np.nanmedian(lefteye_x) - 2 * np.nanstd(lefteye_x))) and currentFrame > 125:
                    #     print('LE Too right: ' + str(currentFrame))
                    #     break
                    # elif ey1 < (np.nanmedian(lefteye_y) - 2 * np.nanstd(lefteye_y)) and currentFrame > 125:
                    #     print('LE Too low: ' + str(currentFrame))
                    #     break

                    # lefteye_sqr_area[currentFrame % 125] = (ew1 * eh1) / 4
                    # lefteye_x[currentFrame % 125] = ex1 + 1 / 4 * ew1
                    # lefteye_y[currentFrame % 125] = ey1 + 1 / 4 * eh1
                    lefteye_ret_area = (ew1 / 2 + 2 * le_d) * (eh1 / 2 + 2 * le_d)

                    # cv2.circle(crop_gray1, (ex1 + int(1 / 4 * ew1) - le_d, ey1 + int(1 / 4 * eh1) - le_d), 7,
                    #            (255, 255, 255), 2)
                    cv2.rectangle(crop_gray1, (ex1 + int(ew1 / 4) - le_d, ey1 + int(eh1 / 4) - le_d),
                                  (ex1 + int(ew1 * 3 / 4) + le_d, ey1 + int(eh1 * 3 / 4) + le_d), (255, 255, 255),
                                  2)
                    # cv2.rectangle(crop_gray1, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (0, 255, 0), 2)
                    # print(str(np.nanmedian(lefteye_sqr_area))+'_'+str(currentFrame))

                    lefteye_img = cv2.GaussianBlur(lefteye_img, (5, 5), 0)
                    # cv2.imshow('lefteye_img', lefteye_img)

                    lefteye_hist_mean = crop_gray1_hist_mean

                    # try:
                    #     crop_gray1_hist_mean = int(indices[0][0]) + 20
                    # except:
                    #     crop_gray1_hist_mean = parameter_crop_gray1_hist_mean

                    # print("CROP_GRAY_MEAN:", crop_gray1_hist_mean)

                    try:
                        iter_threshold = int(round(crop_gray1_hist_mean / 10))
                    except:
                        iter_threshold = 3

                    ret1, thresh1 = cv2.threshold(lefteye_img, crop_gray1_hist_mean, 255, cv2.THRESH_BINARY)
                    # plt.hist(thresh.ravel(),256)

                    kernel = np.ones((3, 3), np.uint8)

                    closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=iter_threshold)
                    # closing1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    cont_img1 = ~closing1.copy()
                    contours1, hierarchy1 = cv2.findContours(cont_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    cont_img1 = cv2.cvtColor(cont_img1, cv2.COLOR_GRAY2BGR)
                    # print(len(contours))

                    contours1 = sorted(contours1, key=lambda x: len(x), reverse=True)

                    for i in range(len(contours1)):
                        if i % 3 == 0:
                            cv2.drawContours(cont_img1, contours1, i, (255, 0, 0), 2)
                        elif i % 3 == 1:
                            cv2.drawContours(cont_img1, contours1, i, (0, 255, 0), 2)
                        else:
                            cv2.drawContours(cont_img1, contours1, i, (0, 0, 255), 2)

                    # ellipse_tmp_r1 = 0
                    # tmp_last1 = 400

                    cnt1_status = 0
                    for i in range(len(contours1)):
                        # print('Pupil Area', area)
                        if len(contours1[i]) < 5:
                            break

                        # area = cv2.contourArea(cnt)

                        # print('Pupil Area', area,
                        #       (np.nanmedian(lefteye_area_buffer) + 2 * np.nanstd(lefteye_area_buffer),
                        #        (np.nanmedian(lefteye_area_buffer) - 2 * np.nanstd(lefteye_area_buffer))))

                        # if (area > (
                        #         np.nanmedian(lefteye_area_buffer) + 2 * np.nanstd(lefteye_area_buffer)) or area < (
                        #             np.nanmedian(lefteye_area_buffer) - 2 * np.nanstd(
                        #         lefteye_area_buffer))) and currentFrame > 125:
                        #     print("LE area is out of range.")
                        #     break

                        # if area < parameter_area_min or area > 1200:
                        #     continue
                        # print(area)
                        # ellipse = cv2.fitEllipse(cnt)
                        try:
                            ellipse1_tmp = cv2.fitEllipse(contours1[i])
                            area1_tmp = math.pi * ellipse1_tmp[1][0] * ellipse1_tmp[1][1] / 4

                            # print("Trial AREA: ", area1_tmp)
                            # print("Trial RATE: ", ellipse1_tmp[1][0] / ellipse1_tmp[1][1])

                            if area1_tmp < parameter_area1_min or area1_tmp > parameter_area1_max:
                                if i == len(contours1) - 1:
                                    cnt1_status = 1
                                    lefteye_error = "CNT Area ERROR."
                                continue

                            if (ellipse1_tmp[1][0] / ellipse1_tmp[1][1]) > (1 / 0.5) or (
                                    ellipse1_tmp[1][0] / ellipse1_tmp[1][1]) < 0.5:
                                if i == len(contours1) - 1:
                                    cnt1_status = 1
                                    lefteye_error = "CNT Rate ERROR."
                                continue

                            ellipse1 = cv2.fitEllipse(contours1[i])
                            area1 = math.pi * ellipse1[1][0] * ellipse1[1][1] / 4
                            lefteye_area = area1

                            # print("Correct AREA: ", area1)
                            # print("Correct RATE: ", ellipse1[1][0] / ellipse1[1][1])

                            break
                        except:
                            break

                        # tmp_x1 = np.array([ellipse_tmp[0][0], 320])
                        # tmp_y1 = np.array([ellipse_tmp[0][1], 240])
                        # tmp_d1 = np.sqrt(np.sum(np.square(tmp_x1 - tmp_y1)))
                        # if (ellipse_tmp[1][0] / ellipse_tmp[1][1] > ellipse_tmp_r1) and (tmp_last1 > tmp_d1):
                        #     ellipse_tmp_r1 = ellipse_tmp[1][0] / ellipse_tmp[1][1]
                        #     ellipse1 = cv2.fitEllipse(cnt)
                        #     tmp_x1 = np.array([ellipse[0][0], 320])
                        #     tmp_y1 = np.array([ellipse[0][1], 240])
                        #     tmp_last1 = np.sqrt(np.sum(np.square(tmp_x1 - tmp_y1)))
                        #
                        # else:
                        #     ellipse_tmp_r1 = ellipse_tmp_r1
                        #     tmp_last1 = tmp_last1
                        # break

                    # cv2.ellipse(lefteye_img, ellipse, (255,0,0), 2)
                    try:
                        if cnt1_status == 1:
                            center1 = last_pos1
                            cv2.ellipse(crop_gray1, center1, (255, 0, 0), 2)
                            center1 = center1[0]
                            cv2.circle(crop_gray1, (int(center1[0]), int(center1[1])), 5, (255, 0, 0), -1)
                            # print('LE CNT ERROR: ' + str(currentFrame))
                            break
                        else:
                            center_tmp1 = ellipse1[0]
                            # center = (center_tmp[0] + ex1, center_tmp[1] + ey1)
                            center1 = (
                                (center_tmp1[0] + ex1 + 1 / 4 * ew1 - le_d, center_tmp1[1] + ey1 + 1 / 4 * eh1 - le_d),
                                ellipse1[1],
                                ellipse1[2])
                            last_pos1 = center1
                            cv2.ellipse(crop_gray1, center1, (255, 0, 0), 2)
                            center1 = center1[0]
                            cv2.circle(crop_gray1, (int(center1[0]), int(center1[1])), 5, (255, 0, 0), -1)
                            lefteye_center_x = int(center1[0])
                            lefteye_center_y = int(center1[1])
                            break
                    except:
                        break

            # print("LEFTEYE CENTER: ", center1[0], center1[1])
            center_x1 = -((center1[0] + 640 - 960) * 0.05 / 12 * 180 / math.pi)
            center_y1 = -((center1[1] - 240) * 0.05 / 12 * 180 / math.pi)

            ##### Show results with pupil center #####
            # combine_thresh = np.hstack((thresh,thresh1))
            # combine_closing = np.hstack((closing,closing1))
            combine = np.hstack((crop_gray, crop_gray1))

            # cv2.imshow setting
            # try:
            #     cv2.imshow('Threshold', thresh)
            #     cv2.imshow('Closing', closing)
            #     cv2.imshow('Counters', cont_img)
            #
            #     cv2.imshow('Threshold1', thresh1)
            #     cv2.imshow('Closing1', closing1)
            #     cv2.imshow('Counters1', cont_img1)
            #
            #     cv2.imshow('Neurobit', combine)
            # except:
            #     print("CV2 occured ERROR.")

            # write the USB stream frame
            combine = cv2.cvtColor(combine, cv2.COLOR_GRAY2BGR)
            out1.write(combine)

            # write pupil center into CSV file for Nystagmus analysis
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d' + 'T' + '%H:%M:%S.%f')[:-3] + 'Z'
            center_x = round(center_x, 3)
            center_y = round(center_y, 3)
            center_x1 = round(center_x1, 3)
            center_y1 = round(center_y1, 3)

            # if currentFrame > 25:
            with open(target_path_csv_record, 'a', newline='') as myFile:
                writer = csv.writer(myFile)
                writer.writerow([st, center_x, center_y, center_x1, center_y1, righteye_area, lefteye_area,
                                 righteye_center_x, righteye_center_y,
                                 lefteye_center_x, lefteye_center_y,
                                 righteye_ret_area, lefteye_ret_area,
                                 righteye_hist_mean, lefteye_hist_mean, righteye_error, lefteye_error])

            # append all time series data in the matrix
            st_All.append(st)
            center_x_All.append(center_x)
            center_y_All.append(center_y)
            center_x1_All.append(center_x1)
            center_y1_All.append(center_y1)

            # generate simulated nystamus type detection
            type1 = type1 + random.randint(0, 1) / 3 + 0.001
            type2 = type2 + random.randint(0, 1) / 3
            type3 = type3 + random.randint(0, 1) * 2
            type4 = type4 + random.randint(0, 1) / 3
            type5 = type5 + random.randint(0, 1) / 2
            type6 = type6 + random.randint(0, 1) / 1
            Brain_risk = (type3 + type5 + type6) / (type1 + type2 + type3 + type4 + type5 + type6)

            # if press keyboard 'q' to stop while loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # To stop duplicate images
            currentFrame += 1

        else:
            break


for i in range(0, len(lc.patient_test_data)):
    target_path = main_path
    try:
        os.chdir(target_path)
        createFolder(lc.patient_test_data[i]['file_name']) # create the dir with the name of "test_ID"
        target_path = os.path.join(target_path, lc.patient_test_data[i]['file_name'])
        os.chdir(target_path)
        # fpath = fpath+'/'+fnameonly[0]
        fall = lc.patient_test_csv[i].replace("csv", file_type)
        # Playing video from file:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        per = "(" + str(i+1) + "/" + str(len(lc.patient_test_data)) + ") "
        print(per + "Now processing video: " + fall + " | test ID: " + lc.patient_test_data[i]['file_name'])
        logs = logs + per + "Now processing video: " + fall + " | test ID: " + lc.patient_test_data[i]['file_name'] + "\n"
        cap = cv2.VideoCapture(fall)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        out = cv2.VideoWriter(lc.patient_test_data[i]['file_name']+"_raw.avi" ,fourcc, 30.0, (1280,480))
        out1 = cv2.VideoWriter(lc.patient_test_data[i]['file_name']+"_f.avi" ,fourcc, 30.0, (1280,480))
        #out = cv2.VideoWriter('file name', 'codec format', 'fps', 'image pixel size')

        eyetrack_all(i)

        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        duration = time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))

        record[i]['duration'] = duration
        record[i]['file_name'] = fall

        print("Duration: %d seconds" % duration)
        print("===============================================")

        logs = logs + "Duration: %d seconds" % duration + "\n"
        logs = logs + "===============================================" + "\n"
    except:
        e_logs = traceback.format_exc()
        print("eye_track.py: " + e_logs)
        logs_name = os.path.join(lc.error_path, "eye_track.txt")
        fp = open(logs_name, "w")
        fp.write(e_logs)
        fp.close
        print("eye_track.py occured ERROR.")

        logs = logs + "eye_track.py: " + e_logs + "\n"
        logs = logs + "eye_track.py occured ERROR." + "\n"

# LOGS
logs_N = os.path.join(lc.error_path, "LOGS.txt")
lp = open(logs_N, "a")
lp.write(logs)
lp.close

# signal processing
def outliers_modified_z_score(ys):
    
    threshold = 0.6745

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold, 1, 0)

def SPV_extract(input):
    
    # compute velocity
    velocity = np.append(0, np.diff(input) * 30 )
    # define SPV feature
    input_out = outliers_modified_z_score(velocity)
    # find SPV index as 0 and 1
    for i in range(1, (np.size(input_out)-2)):
        if (input_out[i-1] & input_out[i+1]) == 1:
            input_out[i] = 1
        elif (input_out[i-1] | input_out[i+1]) == 0:
            input_out[i] = 0
        else:
            input_out[i] = input_out[i]
    input_out1 = input_out
    # find SPV index
    tmp = np.where(input_out1==0)
    index = tmp[0]
    # compute SPV mean std value
    value = []
    for i in range(len(index)):
        value.append( int(velocity[index[i]]) )
    mean = np.mean(value)
    std = np.std(value)
    return velocity, index, value, mean, std


# center_x_v_All, center_x_All_SPV_index, center_x_All_SPV_value, center_x_All_SPV_value_mean, center_x_All_SPV_value_std = SPV_extract(center_x_All)
# center_y_v_All, center_y_All_SPV_index, center_y_All_SPV_value, center_y_All_SPV_value_mean, center_y_All_SPV_value_std = SPV_extract(center_y_All)
# center_x1_v_All, center_x1_All_SPV_index, center_x1_All_SPV_value, center_x1_All_SPV_value_mean, center_x1_All_SPV_value_std = SPV_extract(center_x1_All)
# center_y1_v_All, center_y1_All_SPV_index, center_y1_All_SPV_value, center_y1_All_SPV_value_mean, center_y1_All_SPV_value_std = SPV_extract(center_y1_All)
#
# center_x_All_SPV_value_mean = np.array(center_x_All_SPV_value_mean, dtype='float16')
# center_x_All_SPV_value_std = np.array(center_x_All_SPV_value_std, dtype='float16')
# center_y_All_SPV_value_mean = np.array(center_y_All_SPV_value_mean, dtype='float16')
# center_y_All_SPV_value_std = np.array(center_y_All_SPV_value_std, dtype='float16')
# center_x1_All_SPV_value_mean = np.array(center_x1_All_SPV_value_mean, dtype='float16')
# center_x1_All_SPV_value_std = np.array(center_x1_All_SPV_value_std, dtype='float16')
# center_y1_All_SPV_value_mean = np.array(center_y1_All_SPV_value_mean, dtype='float16')
# center_y1_All_SPV_value_std = np.array(center_y1_All_SPV_value_std, dtype='float16')
#
# center_x_v_All = signal.medfilt(center_x_v_All,21)
# center_y_v_All = signal.medfilt(center_y_v_All,21)
# center_x1_v_All = signal.medfilt(center_x1_v_All,21)
# center_y1_v_All = signal.medfilt(center_y1_v_All,21)

