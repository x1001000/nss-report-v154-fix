# ID setting
# patient_UID = input("Please Enter Patient_UID (ex. D123456789): ")

import os
from os import walk
from os.path import join
import pandas as pd
from numpy import *

import traceback

# select dir_path
import tkinter as tk
from tkinter import filedialog

# record logs
logs = ""

# set data_path
#data_path = os.getcwd()
data_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
re_path = os.path.join(data_path, 'Report')
error_path = os.path.join(re_path, "Error_logs")

print(data_path)
# ID setting
f = open(os.path.join(data_path, 'ID.txt'), 'r')
patient_ID = ""
patient_ID = patient_ID.join(f.readlines())
patient_ID = patient_ID[0:-1] # APP Mode

logs = logs + patient_ID + "\n"
f.close

## APP Mode (If run the Python code without the APP, plz mark)
# data_path = os.path.join(data_path, 'Report')
# os.chdir(data_path)
# error_path = os.path.join(data_path, "Error_logs")

# initialize error_logs
def remove_err(file_name, logs):
    try:
        os.remove(file_name)
    except OSError as e:
        # print(e)
        return logs
    else:
        print(file_name + " is deleted successfully")
        logs = logs + file_name + " is deleted successfully" + "\n"
        return logs

eye_track_err = os.path.join(error_path, 'eye_track.txt')
load_csv_err = os.path.join(error_path, 'load_csv.txt')
main_err = os.path.join(error_path, 'main.txt')
SPV_main_err = os.path.join(error_path, 'SPV_main.txt')
LOGS = os.path.join(error_path, 'LOGS.txt')

err = [eye_track_err, load_csv_err, main_err, SPV_main_err, LOGS]

for i in err:
    logs = remove_err(i,logs)

# set data_path
#data_path = data_path.split('Report')[0] + 'Result'
data_path = os.path.join(data_path, 'Result')

# patient_ID = patient_UID.split(" ")[1] # split ID number
# patient_ID = patient_UID21    `1

def find_patient_data_dir(data_path, patient_ID):
    dir = []
    dir_path = []
    # load spenific patient_ID
    for root, dirs, files in walk(data_path):
        for d in dirs:
            if (patient_ID == d.split(" ")[-1]): # or (patient_ID == d.split("_")[-1]):
                fullpath = join(root, d)
                dir.append(d)
                dir_path.append(fullpath)
            print(patient_ID)
            print(d.split(" ")[-1])    
    
    if len(dir) > 1:
        list = ""
        for i in range(0, len(dir)):
            list += "("
            list += str(i+1)
            list += ") "
            list += str(dir[i])
            list += "\n"

        input_num = input("There are several files regarding to the patient, please enter the number: \n" + list)
        return dir_path[int(input_num)-1], dir[int(input_num)-1]
    return dir_path[0], dir[0]

def find_patient_data_csv(data_path, patient_ID):
    csv_file = []
    file_name = []
    # load specific file_name
    for root, dirs, files in walk(data_path):
      for f in files:
        if (patient_ID in f) and ('.csv' in f):
            if f.replace('.csv', '.avi') in files or f.replace('.csv', '.mp4') in files:
            # print(f)
                file_name.append(f)
                fullpath = join(root, f)
                csv_file.append(fullpath)
    return csv_file, file_name

def find_patient_data_video_type(data_path, patient_ID):
    # load specific file_type (.avi)
    for root, dirs, files in walk(data_path):
      for f in files:
        if (patient_ID in f) and ('.avi' in f):
            return "avi"
        elif (patient_ID in f) and ('.mp4' in f):
            return "mp4"
    return 0

def patient_data(csv_file):
    # create file dic
    file = []
    for i in range(0,len(csv_file)):
        file.append({})

    for i in range(0,len(csv_file)):

        csv_parameter = []

        # use Pandas to deal with data
        df = pd.read_csv(csv_file[i])

        # Find parameters
        file[i]['Date'] = df['Date'][0:1].tolist()
        file[i]['Doctor'] = df['Doctor'][0:1].tolist()
        file[i]['Device'] = df['Device'][0:1].tolist()
        file[i]['Name'] = df['Name'][0:1].tolist()
        file[i]['NIHSS'] = df['NIHSS'][0:1].tolist()
        
        try:
            file[i]['ABCD3i'] = df['ABCD3i'][0:1].tolist() # old version
        except:
            file[i]['ABCD2'] = df['ABCD2'][0:1].tolist()
            
        file[i]['DHI_S'] = df['DHI-S'][0:1].tolist()
        file[i]['Exam'] = df['Exam'][0:1].tolist()
        file[i]['Mode'] = df['Mode'][0:1].tolist()
        
        if file[i]['Mode'][0].split(':')[1][0] == ' ': # naming the name of the test (new version)
            file[i]['file_name'] = file[i]['Mode'][0].split(':')[1][1:] 
        else:
            file[i]['file_name'] = file[i]['Mode'][0].split(':')[1][0:] # old version
            
        file[i]['Speed'] = df['Speed'][0:1].tolist() 
        file[i]['Speed_Mode'] = df['Speed'][0:1].tolist()[0].split(":")[0][-2] # find speed mode
        file[i]['Userdefined_speed'] = df['Userdefined speed'][0:1].tolist()[0]
        file[i]['Sine'] = df['Sine'][0:1].tolist()
        file[i]['Reverse'] = df['Reverse'][0:1].tolist()
        file[i]['Lie'] = df['Lie'][0:1].tolist()

        # Find light spot x,y position
        file[i]['point_x_position'] = df['Doctor'][4:].tolist()
        file[i]['point_x_position'] = list(map(float, file[i]['point_x_position']))
        file[i]['point_y_position'] = df['Device'][4:].tolist()
        file[i]['point_y_position'] = list(map(float, file[i]['point_y_position']))


        # OKN+ or -
        if "OKN" in file[i]['file_name']:
            if str(file[i]['Reverse'][0]) == "True":
                file[i]['file_name'] += "-"
            else:
                file[i]['file_name'] += "+"

        # Fixation and Gaze
        if "Fixation" in file[i]['file_name']: # old version=Fixation, new version=Fixation suppression
            file[i]['file_name'] = "Fixation suppression"

        if "Gaze" in file[i]['file_name']: # old version=Gaze evoked, new version=Gaze
            file[i]['file_name'] = "Gaze"

        # Sin or non-Sin in pursuit
        if "pursuit" in file[i]['file_name']:
            if file[i]['Sine'][0]:
                file[i]['file_name'] = file[i]['file_name'] + " sinusoidal"
            elif not file[i]['Sine'][0]:
                file[i]['file_name'] = file[i]['file_name']

        # Combine mode name and speed number into one name
        if file[i]['Speed_Mode'] != '4':
            if file[i]['Speed'][0].split(':')[1].split('/')[0][0] == ' ':
                sec = file[i]['Speed'][0].split(':')[1].split('/')[0]
                file[i]['file_name'] += sec
            else:    
                file[i]['file_name'] += " "
                sec = file[i]['Speed'][0].split(':')[1].split('/')[0]
                file[i]['file_name'] += sec
        else:
            file[i]['file_name'] = file[i]['file_name'] + " " + str(file[i]['Userdefined_speed']) + "s (" + str(round(1/file[i]['Userdefined_speed'], 2)) + "Hz)"

        # Sin.
        if "pursuit" in file[i]['file_name']:
            if file[i]['Sine'][0]:
                file[i]['file_name'] = file[i]['file_name'] + " (" + str(round(1/float(sec.split("s")[0]), 1)) + "Hz)"

    return file

try:
    patient_test_dir_path, patient_test_dir = find_patient_data_dir(data_path, patient_ID)
    patient_test_csv, patient_test_csv_name = find_patient_data_csv(patient_test_dir_path, patient_ID)

    video_type = find_patient_data_video_type(patient_test_dir_path, patient_ID)
    print("Record video type: " + video_type)
    logs = logs + "Record video type: " + video_type + "\n"
    patient_test_data = patient_data(patient_test_csv)

    print("Successfully load " + str(len(patient_test_csv)) + " CSV Files")
    print("===============================================")
    logs = logs + "Successfully load " + str(len(patient_test_csv)) + " CSV Files" + "\n"
    logs = logs + "===============================================" + "\n"

except:
    e_logs = traceback.format_exc()
    print("load_csv.py: " + e_logs)
    logs = logs + "load_csv.py: " + e_logs + "\n"
    logs_name = os.path.join(error_path, "load_csv.txt")
    fp = open(logs_name, "w")
    fp.write(e_logs)
    fp.close
    print("load_csv.py occured ERROR.")
    logs = logs + "load_csv.py occured ERROR." + "\n"

# LOGS
logs_N = os.path.join(error_path, "LOGS.txt")
lp = open(logs_N, "a")
lp.write(logs)
lp.close






'''
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')
'''



