import re
import cv2
import numpy as np
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
# generate pdf file
from reportlab.pdfgen import canvas  
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import landscape
from reportlab.platypus import Image
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.lib import colors
from reportlab.graphics.shapes import *
from reportlab.graphics import renderPM
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.textlabels import Label
import subprocess

from scipy.signal import find_peaks

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

        # Define the csv and create Nystagmus waveform
        #myFile = open('Pupil_center.csv', 'w')
        target_path_csv_record = os.path.join(target_path, lc.patient_test_data[i]['file_name']) + ".csv"
        with open(target_path_csv_record, 'w', newline='') as myFile:
            myFields = ['Timestamps', 'Righteye_X', 'Righteye_Y', 'Lefteye_X', 'Lefteye_Y']
            writer = csv.DictWriter(myFile, fieldnames=myFields)
            writer.writeheader()

        # Setting the initial value 0 and (0,0) for pupil center and frame image
        # Avoiding none result error
        currentFrame = 0
        ellipse = ((320.0,240.0),(0.0,0.0),0.0)
        ellipse1 = ((960.0,240.0),(0.0,0.0),0.0)
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
        parameter_crop_gray_hist_mean = 35 # code line 130
        parameter_crop_gray1_hist_mean = 35 # code line 196
        parameter_area_min = 1200 # code line 146
        parameter_area1_min = 1200 # code line 212
        parameter_area_max = 20000 # code line 146
        parameter_area1_max = 20000 # code line 212

        # Start to record webcam video forever
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # write the USB stream frame
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                out.write(frame)

            ##### Right eye #####
                crop_gray = gray[0:480, 0:640] # Crop from x, y, w, h -> 100, 200, 300, 400
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

                #plt.hist(crop_gray.ravel(),256)
                #crop_gray_hist = np.histogram(crop_gray.ravel(),256)
                #crop_gray_hist_mean = np.mean(crop_gray_hist[1][0:50])
                crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 0)
                # crop_gray_hist = np.histogram(crop_gray.ravel(),256,[0,256])
                hist = cv2.calcHist([crop_gray], [0], None, [256], [0, 256])
                indices = find_peaks(hist.flatten(), prominence=25, distance=15)
                crop_gray_hist_mean = int(
                    (hist[indices[0][0]] * indices[0][0] + hist[indices[0][1]] * indices[0][1]) / (
                            hist[indices[0][0]] + hist[indices[0][1]]))

                if crop_gray_hist_mean < 15 or crop_gray_hist_mean > 50:
                    crop_gray_hist_mean = parameter_crop_gray_hist_mean
                # if np.mean(crop_gray_hist[0][0:20]) == 0:
                #     crop_gray_hist_mean = 50
                # else:
                #     crop_gray_hist_mean = np.mean(crop_gray_hist[0][20:50])/np.mean(crop_gray_hist[0][0:20]) * 8 + 15

                # if crop_gray_hist_mean < 20:
                #     crop_gray_hist_mean = 20
                # elif crop_gray_hist_mean > 50:
                #     crop_gray_hist_mean = 50
                # crop_gray_hist_mean 20~50 correspond to iter_threshold 8~2
                # 20x+y=8, 50x+y=2, solve x=-1/5, y=12

                iter_threshold = int(round(crop_gray_hist_mean*(-1/5)+12))


                ret,thresh = cv2.threshold(crop_gray,crop_gray_hist_mean,255,cv2.THRESH_BINARY)
                #adap_img = cv2.adaptiveThreshold(crop_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,5) # adaptive threshold
                kernel = np.ones((3, 3), np.uint8)
                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iter_threshold)

                cont_img = ~closing.copy()
                contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                cont_img = cv2.cvtColor(cont_img, cv2.COLOR_GRAY2BGR)
                contours = sorted(contours, key=lambda x: len(x), reverse=True)

                for i in range(len(contours)):
                    if i % 3 == 0:
                        cv2.drawContours(cont_img, contours, i, (255, 0, 0), 3)
                    elif i % 3 == 1:
                        cv2.drawContours(cont_img, contours, i, (0, 255, 0), 3)
                    else:
                        cv2.drawContours(cont_img, contours, i, (0, 0, 255), 3)

                ellipse_tmp_r = 0
                tmp_last = 400
                for cnt in contours:
                    if len(cnt) < 5:
                        break
                    area = cv2.contourArea(cnt)
                    if area < parameter_area_min or area > parameter_area_max:
                        continue
                    #ellipse = cv2.fitEllipse(cnt)
                    ellipse_tmp = cv2.fitEllipse(cnt)
                    tmp_x = np.array([ellipse_tmp[0][0], 320])
                    tmp_y = np.array([ellipse_tmp[0][1], 240])
                    tmp_d = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                    if (ellipse_tmp[1][0]/ellipse_tmp[1][1] > ellipse_tmp_r) and (tmp_last > tmp_d):
                       ellipse_tmp_r = ellipse_tmp[1][0]/ellipse_tmp[1][1]
                       ellipse = cv2.fitEllipse(cnt)
                       tmp_x = np.array([ellipse[0][0], 320])
                       tmp_y = np.array([ellipse[0][1], 240])
                       tmp_last = np.sqrt(np.sum(np.square(tmp_x - tmp_y)))
                    else:
                       ellipse_tmp_r = ellipse_tmp_r
                       tmp_last = tmp_last

                cv2.ellipse(crop_gray, ellipse, (255,0,0), 2)
                center = ellipse[0]
                cv2.circle(crop_gray, (int(center[0]),int(center[1])), 5, (255,0,0), -1)
                #cv2.imshow('Neurobit',crop_gray)

                # classic eyeball diameter is 12 mm (r)
                # length fo Jim's eyes are 24+-1 mm (S)
                # S = r*theta (S:length of eyeball surface, theta: angle of eyeball)
                # left eye amd right eye are 480 pixels = 24 mm (1 pixels = 0.05 mm)
                # theta = pixel*0.05/12*180/pi (degree)
                center_x = -(center[0]-320)*0.05/12*180/math.pi
                center_y = -(center[1]-240)*0.05/12*180/math.pi

            ##### Left eye #####
                crop_gray1 = gray[0:480, 640:1280] # Crop from x, y, w, h -> 100, 200, 300, 400
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

                #plt.hist(crop_gray1.ravel(),256)
                #crop_gray1_hist = np.histogram(crop_gray1.ravel(),256)
                #crop_gray1_hist_mean = np.mean(crop_gray1_hist[1][0:50])
                # crop_gray1_hist = np.histogram(crop_gray1.ravel(),256,[0,256])
                crop_gray1 = cv2.GaussianBlur(crop_gray1, (5, 5), 0)
                hist = cv2.calcHist([crop_gray1], [0], None, [256], [0, 256])

                # indices = find_peaks(hist.flatten(), height=100, distance=15)
                indices = find_peaks(hist.flatten(), prominence=25, distance=15)

                crop_gray1_hist_mean = int((hist[indices[0][0]] * indices[0][0] + hist[indices[0][1]] * indices[0][1]) / (hist[indices[0][0]] + hist[indices[0][1]]))
                #print(np.mean(crop_gray_hist[0][20:50]),np.mean(crop_gray_hist[0][0:20]),np.mean(crop_gray1_hist[0][20:50]),np.mean(crop_gray1_hist[0][0:20]),np.mean(np.histogram(crop_gray.ravel(),256)[1][0:50]),np.mean(np.histogram(crop_gray1.ravel(),256)[1][0:50]))
                if crop_gray1_hist_mean < 15 or crop_gray1_hist_mean > 50:
                    crop_gray1_hist_mean = parameter_crop_gray1_hist_mean

                iter_threshold1 = int(round(crop_gray1_hist_mean*(-1/5)+12))

                # crop_gray1_hist_mean = parameter_crop_gray1_hist_mean

                ret1,thresh1 = cv2.threshold(crop_gray1,crop_gray1_hist_mean,255,cv2.THRESH_BINARY)
                #adap_img = cv2.adaptiveThreshold(crop_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,5) # adaptive threshold
                kernel = np.ones((3, 3), np.uint8)
                closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=iter_threshold1)

                cont_img1 = ~closing1.copy()

                contours1, hierarchy1 = cv2.findContours(cont_img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cont_img1 = cv2.cvtColor(cont_img1, cv2.COLOR_GRAY2BGR)

                contours1 = sorted(contours1, key=lambda x: len(x), reverse=True)

                for i in range(len(contours1)):
                    if i % 3 == 0:
                        cv2.drawContours(cont_img1, contours1, i, (255, 0, 0), 2)
                    elif i % 3 == 1:
                        cv2.drawContours(cont_img1, contours1, i, (0, 255, 0), 2)
                    else:
                        cv2.drawContours(cont_img1, contours1, i, (0, 0, 255), 2)

                ellipse_tmp_r1 = 0
                tmp_last1 = 400
                for cnt in contours1:
                    if len(cnt) < 5:
                        break
                    area = cv2.contourArea(cnt)
                    if area < parameter_area1_min or area > parameter_area1_max:
                        continue
                    #ellipse1 = cv2.fitEllipse(cnt)
                    ellipse_tmp = cv2.fitEllipse(cnt)
                    tmp_x1 = np.array([ellipse_tmp[0][0], 320])
                    tmp_y1 = np.array([ellipse_tmp[0][1], 240])
                    tmp_d1 = np.sqrt(np.sum(np.square(tmp_x1 - tmp_y1)))
                    if (ellipse_tmp[1][0]/ellipse_tmp[1][1] > ellipse_tmp_r1) and (tmp_last1 > tmp_d1):
                       ellipse_tmp_r1 = ellipse_tmp[1][0]/ellipse_tmp[1][1]
                       ellipse1 = cv2.fitEllipse(cnt)
                       tmp_x1 = np.array([ellipse1[0][0], 320])
                       tmp_y1 = np.array([ellipse1[0][1], 240])
                       tmp_last1 = np.sqrt(np.sum(np.square(tmp_x1 - tmp_y1)))
                    else:
                       ellipse_tmp_r1 = ellipse_tmp_r1
                       tmp_last1 = tmp_last1

                cv2.ellipse(crop_gray1, ellipse1, (255,0,0), 2)
                center1 = ellipse1[0]
                cv2.circle(crop_gray1, (int(center1[0]),int(center1[1])), 5, (255,0,0), -1)
                center_x1 = -(center1[0]-320)*0.05/12*180/math.pi
                center_y1 = -(center1[1]-240)*0.05/12*180/math.pi

            ##### Show results with pupil center #####
                combine_thresh = np.hstack((thresh,thresh1))
                combine_closing = np.hstack((closing,closing1))
                combine_cont = np.hstack((cont_img, cont_img1))
                combine = np.hstack((crop_gray,crop_gray1))

                # cv2.imshow setting
                #cv2.imshow('Threshold',combine_thresh)
                #cv2.imshow('Closing',combine_closing)
                #cv2.imshow('Contours', combine_cont)
                #cv2.imshow('Neurobit',combine)

                # write the USB stream frame
                combine = cv2.cvtColor(combine, cv2.COLOR_GRAY2BGR)
                out1.write(combine)

                # write pupil center into CSV file for Nystagmus analysis
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d'+'T'+'%H:%M:%S.%f')[:-3]+'Z'
                center_x = round(center_x,3)
                center_y = round(center_y,3)
                center_x1 = round(center_x1,3)
                center_y1 = round(center_y1,3)

                with open(target_path_csv_record, 'a', newline='') as myFile:
                    writer = csv.writer(myFile)
                    writer.writerow([st, center_x, center_y, center_x1, center_y1])

                # append all time series data in the matrix
                st_All.append(st)
                center_x_All.append(center_x)
                center_y_All.append(center_y)
                center_x1_All.append(center_x1)
                center_y1_All.append(center_y1)

                # generate simulated nystamus type detection
                type1 = type1 + random.randint(0,1)/3+0.001
                type2 = type2 + random.randint(0,1)/3
                type3 = type3 + random.randint(0,1)*2
                type4 = type4 + random.randint(0,1)/3
                type5 = type5 + random.randint(0,1)/2
                type6 = type6 + random.randint(0,1)/1
                Brain_risk = (type3+type5+type6)/(type1+type2+type3+type4+type5+type6)

                #if press keyboard 'q' to stop while loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # To stop duplicate images
                currentFrame += 1

            else:
                break

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


center_x_v_All, center_x_All_SPV_index, center_x_All_SPV_value, center_x_All_SPV_value_mean, center_x_All_SPV_value_std = SPV_extract(center_x_All)
center_y_v_All, center_y_All_SPV_index, center_y_All_SPV_value, center_y_All_SPV_value_mean, center_y_All_SPV_value_std = SPV_extract(center_y_All)
center_x1_v_All, center_x1_All_SPV_index, center_x1_All_SPV_value, center_x1_All_SPV_value_mean, center_x1_All_SPV_value_std = SPV_extract(center_x1_All)
center_y1_v_All, center_y1_All_SPV_index, center_y1_All_SPV_value, center_y1_All_SPV_value_mean, center_y1_All_SPV_value_std = SPV_extract(center_y1_All)

center_x_All_SPV_value_mean = np.array(center_x_All_SPV_value_mean, dtype='float16')
center_x_All_SPV_value_std = np.array(center_x_All_SPV_value_std, dtype='float16')
center_y_All_SPV_value_mean = np.array(center_y_All_SPV_value_mean, dtype='float16')
center_y_All_SPV_value_std = np.array(center_y_All_SPV_value_std, dtype='float16')
center_x1_All_SPV_value_mean = np.array(center_x1_All_SPV_value_mean, dtype='float16')
center_x1_All_SPV_value_std = np.array(center_x1_All_SPV_value_std, dtype='float16')
center_y1_All_SPV_value_mean = np.array(center_y1_All_SPV_value_mean, dtype='float16')
center_y1_All_SPV_value_std = np.array(center_y1_All_SPV_value_std, dtype='float16')

center_x_v_All = signal.medfilt(center_x_v_All,21)
center_y_v_All = signal.medfilt(center_y_v_All,21)
center_x1_v_All = signal.medfilt(center_x1_v_All,21)
center_y1_v_All = signal.medfilt(center_y1_v_All,21)

