import os
from os import walk
from os.path import join

import csv
import pandas as pd

data_path = "D:\TTCH\Result"


def change_name(data_path):
    for root, dirs, files in walk(data_path):
        for d in dirs:
            if len(d.split(" ")[1]) == 10 or len(d.split(" ")[1]) == 11:
                d_name = d.split(" ")[1]
                dir_path = os.path.join(data_path, d)
                print(d_name)
                for root, dirs, files in walk(dir_path):
                    for f in files:
                        if "Patient#_Label_Positive" in f:
                            f_name = f.replace("Patient#_Label_Positive", d_name)
                            o_name, n_name = os.path.join(dir_path, f), os.path.join(dir_path, f_name)
                            print(o_name, n_name)
                            # try:
                            #     os.rename(o_name, n_name)
                            #     print("Finished.")
                            # except:
                            #     print("Failed.")
                            try:
                                os.rename(o_name, n_name)
                            except:
                                print("F.")


                        if "A000000000" in f:
                            f_name = f.replace("A000000000", d_name)
                            o_name, n_name = os.path.join(dir_path, f), os.path.join(dir_path, f_name)
                            print(o_name, n_name)
                            # try:
                            #     os.rename(o_name, n_name)
                            #     print("Finished.")
                            # except:
                            #     print("Failed.")
                            os.rename(o_name, n_name)


def change_csv_patient_name(data_path):
    for root, dirs, files in walk(data_path):
        for d in dirs:
            if len(d.split(" ")[1]) == 10 or len(d.split(" ")[1]) == 11:
                d_name = d.split(" ")[1]
                dir_path = os.path.join(data_path, d)
                print(d_name)
                for root, dirs, files in walk(dir_path):
                    for f in files:
                        if ".csv" in f and d_name in f:
                            print(f)

                            csv_file = os.path.join(dir_path, f)

                            data = pd.read_csv(csv_file)


                            # data['Name'][0] = d_name
                            # data.to_csv(csv_file, index=False)

def find_dir(data_path):
    dir = []
    for root, dirs, files in walk(data_path):
        for d in dirs:
            try:
                if len(d.split(" ")[1]) == 10 or len(d.split(" ")[1]) == 11:
                    print(d)
                    dir.append(d)
            except:
                pass
    return dir

def add_header(data_path):
    pass

# change_name(data_path)
# change_csv_patient_name(data_path)
d = find_dir(data_path)
print(d)


