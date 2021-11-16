# -*- coding: utf-8 -*-
# self-plugins
import load_csv as lc

# packages
import os
import math
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy.signal import lfilter
from scipy.signal import find_peaks
from scipy.signal import filtfilt

import traceback

### parameters
patient_ID = lc.patient_ID
file_path = os.path.join(lc.patient_test_dir_path, "Report")
logs = ""

# mod_x, mod_y, Fs
r_mod_x = 0
r_mod_y = 0
l_mod_x = 0
l_mod_y = 0
Fs = 25

### data crawler
def findout_csv_path(test_ID):
    test_path = os.path.join(file_path, test_ID)
    os.chdir(test_path)
    data_path = os.path.join(test_path, test_ID) + ".csv"
    #print("Now loading : " + data_path)
    return data_path

def righteye_angle_data(data_path):
    df = pd.read_csv(data_path)
    R_X = df['Righteye_X'].tolist()
    R_Y = df['Righteye_Y'].tolist()
    return R_X, R_Y

def lefteye_angle_data(data_path):
    df = pd.read_csv(data_path)
    Lefteye_X = df['Lefteye_X'].tolist()
    Lefteye_Y = df['Lefteye_Y'].tolist()
    return Lefteye_X, Lefteye_Y

def timeline(length):
    Time = []
    for j in range(0, len(length)):
        Time.append(round(j/Fs,2))
    Time = np.asarray(Time)
    return Time

def x_radius_to_angle(r): # +- 20
    angle = math.atan(((r-906)/2348)) * 360 / (2*math.pi)
    return angle

def y_radius_to_angle(r): # +- 10
    angle = -math.atan(((r-486)/2757)) * 360 / (2*math.pi)
    return angle

### SPV Calculation
def round_two(data):
    data = round(data, 2)
    return data

def isoutlier(ys):
    c = 1.4826  # c=-1/(sqrt(2)*erfcinv(3/2))
    MAD = c * np.median(abs(ys - np.median(ys)))  # MAD = c*median(abs(A-median(A)))
    outlier_idx = [x for x in ys if (x > 3 * MAD)]

    return outlier_idx

def z_score(ys):
    threshold = 1  # 0.6745 for modified version
    mean_y = np.mean(ys)
    num_y = len(ys)
    var_y = [np.power((y - mean_y), 2) / num_y for y in ys]
    std_y = [np.sqrt(y) for y in var_y]
    z_scores = [threshold * (y - mean_y) / std_y for y in ys]

    return z_scores


def modified_z_score(ys):
    c1 = 1.253  # When MAD is equal to ZERO.
    c2 = 1.486  # When MAD isn't equal to ZERO.
    median_y = np.median(ys)
    MAD_y = c2 * np.median(abs(ys - median_y))

    modified_z_score = [(y - median_y) / MAD_y for y in ys]

    return modified_z_score

def Nystamus_extract(y, Fs, logs):
    try:
        ## Preprocessing stage
        # y = signal.medfilt(y, 3)
        y_1 = y - np.mean(y)

        out = np.where(abs(y_1) > 20)

        for x in out[0]:
            if x > 0:
                y_1[x] = y_1[x - 1]

        y_2 = modified_z_score(y_1)  # normalized amplitude
        # y_3 = signal.medfilt(y_2, 11)  # median filter

        # Load filter parameter
        FIR1 = np.array(
            [-0.0296451204833518, 0.00925172607229440, -0.0115293989022348, 0.0140375254341020, -0.0167393289436908,
             0.0195876175466524, -0.0225259190063055, 0.0254901124852358, -0.0284104995644265, 0.0312142309050116,
             -0.0338279828852039, 0.0361807618278533, -0.0382067031641653, 0.0398477298031179, -0.0410559384117443,
             0.0417955941208891, 1.00907118633193, 0.0417955941208891, -0.0410559384117443, 0.0398477298031179,
             -0.0382067031641653, 0.0361807618278533, -0.0338279828852039, 0.0312142309050116, -0.0284104995644265,
             0.0254901124852358, -0.0225259190063055, 0.0195876175466524, -0.0167393289436908, 0.0140375254341020,
             -0.0115293989022348, 0.00925172607229440, -0.0296451204833518])
        FIR2 = np.array(
            [0.0126790233155853, 0.00260959042439373, 0.00357784368011279, 0.00457039817392485, 0.00553924417565522,
             0.00642922548646735, 0.00717965385859548, 0.00772634931826725, 0.00800404249455777, 0.00794905231651221,
             0.00750213284050330, 0.00661136771684943, 0.00523498092281557, 0.00334392865357511, 0.000924140148254810,
             -0.00202171512828932, -0.00547304034574246, -0.00939078846528278, -0.0137176607555862, -0.0183790526975593,
             -0.0232847398817529, -0.0283312622978241, -0.0334049321297642, -0.0383853594960468, -0.0431493641058940,
             -0.0475751199533050, -0.0515463660886133, -0.0549565100146286, -0.0577124517959393, -0.0597379665859794,
             -0.0609765005955555, 0.961827736572315, -0.0609765005955555, -0.0597379665859794, -0.0577124517959393,
             -0.0549565100146286, -0.0515463660886133, -0.0475751199533050, -0.0431493641058940, -0.0383853594960468,
             -0.0334049321297642, -0.0283312622978241, -0.0232847398817529, -0.0183790526975593, -0.0137176607555862,
             -0.00939078846528278, -0.00547304034574246, -0.00202171512828932, 0.000924140148254810, 0.00334392865357511,
             0.00523498092281557, 0.00661136771684943, 0.00750213284050330, 0.00794905231651221, 0.00800404249455777,
             0.00772634931826725, 0.00717965385859548, 0.00642922548646735, 0.00553924417565522, 0.00457039817392485,
             0.00357784368011279, 0.00260959042439373, 0.0126790233155853])
        FIR3 = np.array(
            [0.0166161054134519, -0.00210022371807598, 0.00177986913220994, -0.00133019530735843, 0.000740011376541466,
             -1.91201881788006e-17, -0.000896901583959258, 0.00195515458409058, -0.00317631732882165, 0.00455880033204366,
             -0.00609768730834985, 0.00778463016723917, -0.00960782418992972, 0.0115520675165402, -0.0135989068041230,
             0.0157268685236960, -0.0179117729190226, 0.0201271252267365, -0.0223445764326939, 0.0245344436876570,
             -0.0266662785968899, 0.0287094699966587, -0.0306338665908435, 0.0324104039869497, -0.0340117202744467,
             0.0354127443476540, -0.0365912416941331, 0.0375283033368980, -0.0382087650095708, 0.0386215454190648,
             0.930237469421573, 0.0386215454190648, -0.0382087650095708, 0.0375283033368980, -0.0365912416941331,
             0.0354127443476540, -0.0340117202744467, 0.0324104039869497, -0.0306338665908435, 0.0287094699966587,
             -0.0266662785968899, 0.0245344436876570, -0.0223445764326939, 0.0201271252267365, -0.0179117729190226,
             0.0157268685236960, -0.0135989068041230, 0.0115520675165402, -0.00960782418992972, 0.00778463016723917,
             -0.00609768730834985, 0.00455880033204366, -0.00317631732882165, 0.00195515458409058, -0.000896901583959258,
             -1.91201881788006e-17, 0.000740011376541466, -0.00133019530735843, 0.00177986913220994, -0.00210022371807598,
             0.0166161054134519])

        ## Use lfilter to filter x with the FIR filter.
        # y_4 = filtfilt(FIR1, 1,
        #               y_3)  # The low-pass filtering with fcut-off = 30 Hz realized as the 32th order low-pass FIR filter.
        # y_5 = filtfilt(FIR2, 1,
        #               y_4)  # The high-pass filtering with fcut-off = 1.5 Hz applying the Chebyshev window with 20 dB of relative sidelobe attenuation is also used. The order of the filter is 62.
        # y_6 = filtfilt(FIR3, 1,
        #               y_5)  # The low-pass FIR filtering with fcut-off = 25 Hz realized as the 60th order low-pass FIR filter and the Chebyshev window with 20 dB of relative sidelobe attenuation is also used.

        # low frame-rate
        y_3 = 0 # low frame-rate
        y_4 = 0 # low frame-rate
        y_5 = 0 # low frame-rate
        y_6 = 0 # low frame-rate

        ## Non-linear operation
        y_7 = np.power(np.diff(y_2), 2)

        ## Peak detection
        # saccade last as high as 350 ms / mean is 250 ms
        peaks, properties = find_peaks(y_7, prominence=0.8, distance= 6) # distance = 250 / (1000/Fs)
        pks = properties.get('prominences')
        locs = peaks

        # remove noise peak
        # c = 1.4826  # c=-1/(sqrt(2)*erfcinv(3/2))
        # MAD = c * np.median(abs(pks - np.median(pks)))  # MAD = c*median(abs(A-median(A)))
        # outlier_val = [x for x in pks if (x > 3 * MAD)]

        tmp1 = []
        # for i in range(len(outlier_val)):
        #     tmp = np.argwhere(pks == outlier_val[i])
        #     tmp1 = np.append(tmp1, tmp)
        # tmp1 = tmp1.astype(int)

        locs_f = np.delete(locs, tmp1)
        pks_f = np.delete(pks, tmp1)

        # y_f = []
        # for j in range(0, len(locs_f)):
        #      y_f.append(y[locs_f[j]])
        #
        # y_2_f = []
        # for j in range(0, len(locs_f)):
        #     y_2_f.append(y_2[locs_f[j]])
        #
        # y_7_f = []
        # for j in range(0, len(locs_f)):
        #     y_7_f.append(y_7[locs_f[j]])

        return locs, pks, y_1, locs, logs
    except:
        locs = pks = y_1 = locs = "N/A"
        print("Nystagmus_extract occured ERROR.")
        logs = logs + "Nystagmus_extract occured ERROR." + "\n"
        return locs, pks, y_1, locs, logs

def SPV_extract(input, Fs, logs):
    try:
        # define SPV feature
        interval = 1 / Fs
        # input = signal.medfilt(input, 11)
        # input = signal.medfilt(input, 3)
        y_1 = input - np.mean(input)

        out = np.where(abs(y_1) > 20)

        for x in out[0]:
            if x > 0:
                y_1[x] = y_1[x-1]

        # try:
        #     for i in range(len(y_1)):
        #         if (y_1[i] > 15 or y_1[i] < -15) and i > 0:
        #             y_1[i] = y_1[i - 1]
        # except:
        #     print("SPV FILTER GOT WORNG.")

        # find out SPV index (non_zero value)
        input_1 = np.diff(y_1) / interval
        zero_idx = np.where(input_1 == 0)
        input_nonzero = np.delete(input_1, zero_idx)

        # MAD calculation (=95% confidence interval)
        c = 1.4826  # c=-1/(sqrt(2)*erfcinv(3/2))
        MAD = c * np.median(abs(input_nonzero - np.median(input_nonzero)))  # MAD = c*median(abs(A-median(A)))
        input_2 = (abs(input_nonzero - np.median(input_nonzero)) / MAD) > 1 # mod?
        input_idx = input_2 # record value before filtering

        # find SPV index as 0 and 1 (noise of fast phase)
        for i in range(1, (len(input_2) - 2)):
            if (input_2[i - 1] & input_2[i + 1]) == 1:
                input_2[i] = 1
            elif (input_2[i - 1] | input_2[i + 1]) == 0:
                input_2[i] = 0
            else:
                input_2[i] = input_2[i]
        input_3 = ~input_2

        # find SPV index
        for i in range(0, len(zero_idx[0])): # add zero-idx back
            input_3 = np.insert(input_3, zero_idx[0][i], 0)
        tmp = np.where(input_3 == 1)
        index = tmp[0]
        y_2 = input_nonzero
        y_2 = signal.medfilt(input_nonzero, 11)
        y_idx = y_2 # record value before add-back
        for i in range(0, len(zero_idx[0])): # add zero-idx back
            y_2 = np.insert(y_2, zero_idx[0][i], 0)
        y_3 = []
        out = []

        for i in range(len(index)):
            y_3 = np.append(y_3, y_2[index[i]])
            out = np.append(out, input_1[index[i]])

        # y_4 = [x for x in y_3 if (x > np.mean(y_3) - 2 * np.std(y_3)) & (x < np.mean(y_3) + 2 * np.std(y_3))] # 95% confidence interval
        y_4 = y_3

        # print(y_4)

        # compute SPV mean std value
        mean = np.mean(y_4)
        std = np.std(y_4)
        median = np.median(y_4)
        q1, q3 = np.percentile(y_4, [25, 75])
        iqr = q3 - q1
        # iqr = 0
        return input_1, index, y_4, mean, std, median, iqr, logs
    except:
        input_1 = index = y_4 = mean = std = median = iqr = "N/A"
        print("SPV_extract occured ERROR.")
        logs = logs + "SPV_extract occured ERROR." + "\n"
        return input_1, index, y_4, mean, std, median, iqr, logs

def pursuit_gain(input, input_v, SPV_idx, speed, speed_mode, user_defined, direction, logs):
    try:
        print("SPEED Mode: %s" % speed_mode)

        # if speed_mode == '4':
        #     speed = int(user_defined)
        # else:
        #     speed = int(speed)
        speed = user_defined = float(user_defined)
        print("SPEED: %d" % speed)

        target_velocity_3s = 20  # (15° - (- 15°)) / 1.5 = 20 °/s (Pursuit 3s)
        target_velocity_5s = 12  # (15° - (- 15°)) / 2.5 = 12 °/s (Pursuit 5s)
        target_velocity_10s = 6  # (15° - (- 15°)) / 5 = 6 °/s (Pursuit 10s)
        target_velocity_user = int(30 / user_defined)

        target_velocity_3s_v = 13  # (10° - (- 10°)) / 1.5 = 13 °/s (Pursuit 3s)
        target_velocity_5s_v = 8  # (10° - (- 10°)) / 2.5 = 8 °/s (Pursuit 5s)
        target_velocity_10s_v = 4 # (10° - (- 10°)) / 5 = 4 °/s (Pursuit 10s)
        target_velocity_user_v = int(20 / user_defined)

        SPV = input_v[SPV_idx]

        p_locs = find_peaks(input, height=8, distance=speed / 2 * 25)[0]
        n_locs = find_peaks(-input, height=8, distance=speed / 2 * 25)[0]

        # except
        if len(p_locs) < 2 or len(n_locs) < 2:
            print("pursuit_gain: p_locs = %d, n_lods = %d"  % (len(p_locs), len(n_locs)))
            logs = logs + "pursuit_gain occured ERROR." + "\n"
            p_locs = n_locs = pg_locs = ng_locs = p_gain = n_gain = m_gain = p_med = n_med = m_med = "N/A"
            return p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain, p_med, n_med, m_med, logs

        p_locs = np.asarray(p_locs)
        n_locs = np.asarray(n_locs)

        # print(p_locs, n_locs)


        m_locs = np.sort(np.concatenate((p_locs, n_locs)))

        # print(m_locs)

        pg_locs = []
        ng_locs = []
        idx = 0

        for i in range(0, len(m_locs)):
            if i % 2 == 0 and i != (len(m_locs) - 1):
                for j in range(idx, len(SPV_idx)):
                    if SPV_idx[j] >= m_locs[i] and SPV_idx[j] <= m_locs[i+1]:
                        ng_locs.append(SPV_idx[j])
                    elif SPV_idx[j] > m_locs[i+1]:
                        idx = j
                        break
            elif len(m_locs) == 2:
                for j in range(0, len(SPV_idx)):
                    if SPV_idx[j] >= 0 and SPV_idx[j] <= m_locs[0]:
                        pg_locs.append(SPV_idx[j])
                    elif SPV_idx[j] > m_locs[1]:
                        break
            elif i != (len(m_locs) - 1):
                for j in range(idx, len(SPV_idx)):
                    if SPV_idx[j] >= m_locs[i] and SPV_idx[j] <= m_locs[i+1]:
                        pg_locs.append(SPV_idx[j])
                    elif SPV_idx[j] > m_locs[i+1]:
                        idx = j
                        break

        pg_locs = np.asarray(pg_locs)
        ng_locs = np.asarray(ng_locs)

        # print(m_locs)
        # print(pg_locs)
        # print(ng_locs)

        p_v = input_v[pg_locs]
        n_v = input_v[ng_locs]

        # print(p_v)
        # print(n_v)

        # Horizontal
        if speed == 3 and direction == "Horizontal":
            p_gain = abs(np.mean(p_v)) / target_velocity_3s
            p_med = abs(np.median(p_v)) / target_velocity_3s
            n_gain = abs(np.mean(n_v)) / target_velocity_3s
            n_med = abs(np.median(n_v)) / target_velocity_3s
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2
        elif speed == 5 and direction == "Horizontal":
            p_gain = abs(np.mean(p_v)) / target_velocity_5s
            p_med = abs(np.median(p_v)) / target_velocity_5s
            n_gain = abs(np.mean(n_v)) / target_velocity_5s
            n_med = abs(np.median(n_v)) / target_velocity_5s
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2
        elif speed == 10 and direction == "Horizontal":
            p_gain = abs(np.mean(p_v)) / target_velocity_10s
            p_med = abs(np.median(p_v)) / target_velocity_10s
            n_gain = abs(np.mean(n_v)) / target_velocity_10s
            n_med = abs(np.median(n_v)) / target_velocity_10s
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2
        elif speed == user_defined:
            p_gain = abs(np.mean(p_v)) / target_velocity_user
            p_med = abs(np.median(p_v)) / target_velocity_user
            n_gain = abs(np.mean(n_v)) / target_velocity_user
            n_med = abs(np.median(n_v)) / target_velocity_user
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2


        # Vertical
        elif speed == 3 and direction == "Vertical":
            p_gain = abs(np.mean(p_v)) / target_velocity_10s_v
            p_med = abs(np.median(p_v)) / target_velocity_10s_v
            n_gain = abs(np.mean(n_v)) / target_velocity_10s_v
            n_med = abs(np.median(n_v)) / target_velocity_10s_v
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2
        elif speed == 5 and direction == "Vertical":
            p_gain = abs(np.mean(p_v)) / target_velocity_10s_v
            p_med = abs(np.median(p_v)) / target_velocity_10s_v
            n_gain = abs(np.mean(n_v)) / target_velocity_10s_v
            n_med = abs(np.median(n_v)) / target_velocity_10s_v
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2
        elif speed == 10 and direction == "Vertical":
            p_gain = abs(np.mean(p_v)) / target_velocity_10s_v
            p_med = abs(np.median(p_v)) / target_velocity_10s_v
            n_gain = abs(np.mean(n_v)) / target_velocity_10s_v
            n_med = abs(np.median(n_v)) / target_velocity_10s_v
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2
        elif speed == user_defined:
            p_gain = abs(np.mean(p_v)) / target_velocity_user_v
            p_med = abs(np.median(p_v)) / target_velocity_user_v
            n_gain = abs(np.mean(n_v)) / target_velocity_user_v
            n_med = abs(np.median(n_v)) / target_velocity_user_v
            m_gain = (p_gain + n_gain) / 2
            m_med = (p_med + n_med) / 2

        # print(p_gain, n_gain, m_gain)

        return p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain, p_med, n_med, m_med, logs
    except:
        p_locs = n_locs = pg_locs = ng_locs = p_gain = n_gain = m_gain = p_med = n_med = m_med = "N/A"
        print("pursuit_gain occured ERROR.")
        logs = logs + "pursuit_gain occured ERROR." + "\n"
        return p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain, p_med, n_med, m_med, logs

def saccade_gain(input, input_v, FP_idx, user_defined, logs):
    try:
        speed = user_defined = float(user_defined)
        print("SPEED: %d" % speed)

        p_locs = find_peaks(input, height=8, distance=speed / 2 * 25)
        n_locs = find_peaks(-input, height=8, distance=speed / 2 * 25)

        p_pks = []
        n_pks = []
        for i in range(len(p_locs[0])):
            p_pks.append(input[p_locs[0][i]])

        for i in range(len(n_locs[0])):
            n_pks.append(input[n_locs[0][i]])

        p_pks = np.asarray(p_pks)
        n_pks = np.asarray(n_pks)

        p_first_amp_loc = []
        n_first_amp_loc = []
        p_gain_amp_loc, n_gain_amp_loc = [], []
        p_f_amp = []
        n_f_amp = []

        for i in range(0, len(FP_idx)):
            if input[FP_idx[i]] < 0:
                p_gain_amp_loc.append(FP_idx[i])
                p_first_amp_loc.append(FP_idx[i] + 1)
            elif input[FP_idx[i]] > 0:
                n_gain_amp_loc.append(FP_idx[i])
                n_first_amp_loc.append(FP_idx[i] + 1)

        p_first_amp_loc = np.asarray(p_first_amp_loc)
        n_first_amp_loc = np.asarray(n_first_amp_loc)
        p_gain_amp_loc = np.asarray(p_gain_amp_loc)
        n_gain_amp_loc = np.asarray(n_gain_amp_loc)

        for i in range(len(p_first_amp_loc)):
            p_f_amp.append(input[p_first_amp_loc[i]])
        for i in range(len(n_first_amp_loc)):
            n_f_amp.append(input[n_first_amp_loc[i]])

        p_f_amp = [x for x in p_f_amp if abs((x - np.mean(p_f_amp)) / np.std(p_f_amp)) < 2]
        n_f_amp = [x for x in n_f_amp if abs((x - np.mean(n_f_amp)) / np.std(n_f_amp)) < 2]
        p_g_amp = [x for x in input[p_gain_amp_loc] if abs((x - np.mean(input[p_gain_amp_loc])) / np.std(input[p_gain_amp_loc])) < 2]
        # print(p_g_amp)
        n_g_amp = [x for x in input[n_gain_amp_loc] if abs((x - np.mean(input[n_gain_amp_loc])) / np.std(input[n_gain_amp_loc])) < 2]
        # print(n_g_amp)

        r_gain = (np.mean(p_f_amp) - np.mean(p_g_amp)) / (np.mean(p_pks) - np.mean(p_g_amp))
        l_gain = (np.mean(n_f_amp) - np.mean(n_g_amp)) / (np.mean(n_pks) - np.mean(n_g_amp))
        m_gain = (r_gain + l_gain) / 2

        p_peak_v = (np.mean(p_f_amp) - np.mean(p_g_amp)) / (1 / 25)
        n_peak_v = (np.mean(n_f_amp) - np.mean(n_g_amp)) / (1 / 25)
        m_peak_v = (abs(p_peak_v) + abs(n_peak_v)) / 2

        r_med = (np.median(p_f_amp) - np.median(p_g_amp)) / (np.median(p_pks) - np.median(p_g_amp))
        l_med = (np.median(n_f_amp) - np.median(n_g_amp)) / (np.median(n_pks) - np.median(n_g_amp))
        m_med = (r_gain + l_gain) / 2

        p_med_v = (np.median(p_f_amp) - np.median(p_g_amp)) / (1 / 25)
        n_med_v = (np.median(n_f_amp) - np.median(n_g_amp)) / (1 / 25)
        m_med_v = (abs(p_peak_v) + abs(n_peak_v)) / 2

        return (p_locs[0], n_locs[0], p_first_amp_loc, n_first_amp_loc,
                r_gain, l_gain, m_gain, r_med, l_med, m_med,
                p_peak_v, n_peak_v, m_peak_v, p_med_v, n_med_v, m_med_v, logs)

    except:
        p_first_amp_loc = n_first_amp_loc = "N/A"
        r_gain = l_gain = m_gain = r_med = l_med = m_med = "N/A"
        p_peak_v = n_peak_v = m_peak_v = p_med_v = n_med_v = m_med_v = "N/A"
        print("saccade_gain occured ERROR.")
        logs = logs + "saccade_gain occured ERROR." + "\n"
        return (p_first_amp_loc, n_first_amp_loc,
                r_gain, l_gain, m_gain, r_med, l_med, m_med,
                p_peak_v, n_peak_v, m_peak_v, p_med_v, n_med_v, m_med_v, logs)


def OKN_asymmetry(input):

    return 0

def gaze_SPV(target_H, target_V, input_X, input_Y, logs):
    try:
        # parameters
        center = []
        right = []
        left = []
        down = []
        up = []

        target_H = np.asarray(target_H)
        target_V = np.asarray(target_V)

        target_H = target_H[0:len(input_X)]
        target_V = target_V[0:len(input_Y)]

        # print(target_V, target_H)

        for i in range(0, len(target_H)):
            if (int(target_H[i]) == 0) and (int(target_V[i]) == 0) and (down == []):
                center.append(i)
            elif (int(target_H[i]) > 0) and (int(target_V[i]) == 0) and (down == []):
                right.append(i)
            elif (int(target_H[i]) < 0) and (int(target_V[i]) == 0) and (down == []):
                left.append(i)
            elif (int(target_H[i]) == 0) and (int(target_V[i]) > 0) and (down == []):
                up.append(i)
            elif (int(target_H[i]) == 0) and (int(target_V[i]) < 0):
                down.append(i)
        try:
            for i in range(down[-1]+1, len(target_H)):
                if (int(target_H[i]) == 0) and (int(target_V[i]) == 0):
                    center.append(i)
                elif (int(target_H[i]) > 0) and (int(target_V[i]) == 0):
                    right.append(i)
                elif (int(target_H[i]) < 0) and (int(target_V[i]) == 0):
                    left.append(i)
                elif (int(target_H[i]) == 0) and (int(target_V[i]) > 0):
                    up.append(i)
                elif (int(target_H[i]) == 0) and (int(target_V[i]) < 0):
                    down.append(i)
        except:
            pass

        # center = np.asarray(center)
        # right = np.asarray(right)
        # left = np.asarray(left)
        # down = np.asarray(down)
        # up = np.asarray(up)

        # print(center, right, left, down, up)
        # c = input_X[center]
        # print(c)

        if center:
            center = np.asarray(center)
            c_input_1_H, c_index_H, c_value_H, c_mean_H, c_std_H, c_median_H, c_iqr_H, logs = SPV_extract(input_X[center], 25, logs)
            c_locs, c_pks, c_y_1, c_peaks_H, logs = Nystamus_extract(input_X[center], 25, logs)
            c_saccade_num = len(c_locs)
            c_saccade_num_FR = round(c_saccade_num / (len(center)/25) * 10, 4)

            c_input_1_V, c_index_V, c_value_V, c_mean_V, c_std_V, c_median_V, c_iqr_V, logs = SPV_extract(input_Y[center], 25, logs)
            c_locs_V, c_pks_V, c_y_1_V, c_peaks_V, logs = Nystamus_extract(input_Y[center], 25, logs)
            c_saccade_num_V = len(c_locs_V)
            c_saccade_num_FR_V = round(c_saccade_num_V / (len(center)/25) * 10, 4)
        else:
            c_mean_H = c_std_H = c_median_H = c_iqr_H = c_saccade_num = c_saccade_num_FR = "N/A"
            c_mean_V = c_std_V = c_median_V = c_iqr_V = c_saccade_num_V = c_saccade_num_FR_V = "N/A"

        if right:
            right = np.asarray(right)
            r_input_1_H, r_index_H, r_value_H, r_mean_H, r_std_H, r_median_H, r_iqr_H, logs = SPV_extract(input_X[right], 25, logs)
            r_locs, r_pks, r_y_1, r_peaks_H, logs = Nystamus_extract(input_X[right], 25, logs)
            r_saccade_num = len(r_locs)
            r_saccade_num_FR = round(r_saccade_num / (len(right)/25) * 10, 4)

            r_input_1_V, r_index_V, r_value_V, r_mean_V, r_std_V, r_median_V, r_iqr_V, logs = SPV_extract(input_Y[right], 25, logs)
            r_locs_V, r_pks_V, r_y_1_V, r_peaks_V, logs = Nystamus_extract(input_Y[right], 25, logs)
            r_saccade_num_V = len(r_locs_V)
            r_saccade_num_FR_V = round(r_saccade_num_V / (len(right)/25) * 10, 4)
        else:
            r_mean_H = r_std_H = r_median_H = r_iqr_H = r_saccade_num = r_saccade_num_FR = "N/A"
            r_mean_V = r_std_V = r_median_V = r_iqr_V = r_saccade_num_V = r_saccade_num_FR_V = "N/A"

        if left:
            left = np.asarray(left)
            l_input_1_H, l_index_H, l_value_H, l_mean_H, l_std_H, l_median_H, l_iqr_H, logs = SPV_extract(input_X[left], 25, logs)
            l_locs, l_pks, l_y_1, l_peaks_H, logs = Nystamus_extract(input_X[left], 25, logs)
            l_saccade_num = len(l_locs)
            l_saccade_num_FR = round(l_saccade_num / (len(left)/25) * 10, 4)

            l_input_1_V, l_index_V, l_value_V, l_mean_V, l_std_V, l_median_V, l_iqr_V, logs = SPV_extract(input_Y[left], 25, logs)
            l_locs_V, l_pks_V, l_y_1_V, l_peaks_V, logs = Nystamus_extract(input_Y[left], 25, logs)
            l_saccade_num_V = len(l_locs_V)
            l_saccade_num_FR_V = round(l_saccade_num_V / (len(left)/25) * 10, 4)
        else:
            l_mean_H = l_std_H = l_median_H = l_iqr_H = l_saccade_num = l_saccade_num_FR = "N/A"
            l_mean_V = l_std_V = l_median_V = l_iqr_V = l_saccade_num_V = l_saccade_num_FR_V = "N/A"

        if down:
            down = np.asarray(down)
            d_input_1_Y, d_index_Y, d_value_Y, d_mean_Y, d_std_Y, d_median_Y, d_iqr_Y, logs = SPV_extract(input_Y[down], 25, logs)
            d_locs, d_pks, d_y_1, d_peaks_H, logs = Nystamus_extract(input_Y[down], 25, logs)
            d_saccade_num = len(d_locs)
            d_saccade_num_FR = round(d_saccade_num / (len(down)/25) * 10, 4)

            d_input_1_H, d_index_H, d_Halue_H, d_mean_H, d_std_H, d_median_H, d_iqr_H, logs = SPV_extract(input_X[down], 25, logs)
            d_locs_H, d_pks_H, d_y_1_H, d_peaks_H, logs = Nystamus_extract(input_X[down], 25, logs)
            d_saccade_num_H = len(d_locs_H)
            d_saccade_num_FR_H = round(d_saccade_num_H / (len(down)/25) * 10, 4)
        else:
            d_mean_Y = d_std_Y = d_median_Y = d_iqr_Y = d_saccade_num = d_saccade_num_FR = "N/A"
            d_mean_H = d_std_H = d_median_H = d_iqr_H = d_saccade_num_H = d_saccade_num_FR_H = "N/A"

        if up:
            up = np.asarray(up)
            u_input_1_Y, u_index_Y, u_value_Y, u_mean_Y, u_std_Y, u_median_Y, u_iqr_Y, logs = SPV_extract(input_Y[up], 25, logs)
            u_locs, u_pks, u_y_1, u_peaks_H, logs= Nystamus_extract(input_Y[up], 25, logs)
            u_saccade_num = len(u_locs)
            u_saccade_num_FR = round(u_saccade_num / (len(up)/25) * 10, 4)

            u_input_1_H, u_index_H, u_Halue_H, u_mean_H, u_std_H, u_median_H, u_iqr_H, logs = SPV_extract(input_X[up], 25, logs)
            u_locs_H, u_pks_H, u_y_1_H, u_peaks_H, logs = Nystamus_extract(input_X[up], 25, logs)
            u_saccade_num_H = len(u_locs_H)
            u_saccade_num_FR_H = round(u_saccade_num_H / (len(up)/25) * 10, 4)
        else:
            u_mean_Y = u_std_Y = u_median_Y = u_iqr_Y = u_saccade_num = u_saccade_num_FR = "N/A"
            u_mean_H = u_std_H = u_median_H = u_iqr_H = u_saccade_num_H = u_saccade_num_FR_H = "N/A"

        # c_locs, c_pks, c_y_1, c_peaks_H = Nystamus_extract(input_X[center], 25)
        # r_locs, r_pks, r_y_1, r_peaks_H = Nystamus_extract(input_X[right], 25)
        # l_locs, l_pks, l_y_1, l_peaks_H = Nystamus_extract(input_X[left], 25)
        # d_locs, d_pks, d_y_1, d_peaks_H = Nystamus_extract(input_Y[down], 25)
        # u_locs, u_pks, u_y_1, u_peaks_H = Nystamus_extract(input_Y[up], 25)
        #
        # c_saccade_num = len(c_locs)
        # c_saccade_num_FR = round(c_saccade_num / Time[center[-1]] * 1000, 4)
        #
        # r_saccade_num = len(r_locs)
        # r_saccade_num_FR = round(r_saccade_num / Time[right[-1]] * 1000, 4)
        #
        # l_saccade_num = len(l_locs)
        # l_saccade_num_FR = round(l_saccade_num / Time[left[-1]] * 1000, 4)
        #
        # d_saccade_num = len(d_locs)
        # d_saccade_num_FR = round(d_saccade_num / Time[down[-1]] * 1000, 4)
        #
        # u_saccade_num = len(u_locs)
        # u_saccade_num_FR = round(u_saccade_num / Time[up[-1]] * 1000, 4)

        # print(c_mean_H, c_std_H, c_median_H, c_iqr_H,
        #         c_mean_V, c_std_V, c_median_V, c_iqr_V,
        #         r_mean_H, r_std_H, r_median_H, r_iqr_H,
        #         l_mean_H, l_std_H, l_median_H, l_iqr_H,
        #         d_mean_Y, d_std_Y, d_median_Y, d_iqr_Y,
        #         u_mean_Y, u_std_Y, u_median_Y, u_iqr_Y,
        #         c_saccade_num, c_saccade_num_FR,
        #         c_saccade_num_V, c_saccade_num_FR_V,
        #         l_saccade_num, l_saccade_num_FR,
        #         r_saccade_num, r_saccade_num_FR,
        #         d_saccade_num, d_saccade_num_FR,
        #         u_saccade_num, u_saccade_num_FR)


        return (c_mean_H, c_std_H, c_median_H, c_iqr_H,
                c_mean_V, c_std_V, c_median_V, c_iqr_V,
                r_mean_H, r_std_H, r_median_H, r_iqr_H,
                r_mean_V, r_std_V, r_median_V, r_iqr_V,
                l_mean_H, l_std_H, l_median_H, l_iqr_H,
                l_mean_V, l_std_V, l_median_V, l_iqr_V,
                d_mean_Y, d_std_Y, d_median_Y, d_iqr_Y,
                d_mean_H, d_std_H, d_median_H, d_iqr_H,
                u_mean_Y, u_std_Y, u_median_Y, u_iqr_Y,
                u_mean_H, u_std_H, u_median_H, u_iqr_H,
                c_saccade_num, c_saccade_num_FR,
                c_saccade_num_V, c_saccade_num_FR_V,
                l_saccade_num, l_saccade_num_FR,
                l_saccade_num_V, l_saccade_num_FR_V,
                r_saccade_num, r_saccade_num_FR,
                r_saccade_num_V, r_saccade_num_FR_V,
                d_saccade_num, d_saccade_num_FR,
                d_saccade_num_H, d_saccade_num_FR_H,
                u_saccade_num, u_saccade_num_FR,
                u_saccade_num_H, u_saccade_num_FR_H,
                logs)
    except:
        c_mean_H = c_std_H = c_median_H = c_iqr_H = "N/A"
        c_mean_V = c_std_V = c_median_V = c_iqr_V = "N/A"
        r_mean_H = r_std_H = r_median_H = r_iqr_H = "N/A"
        r_mean_V = r_std_V = r_median_V = r_iqr_V = "N/A"
        l_mean_H = l_std_H = l_median_H = l_iqr_H = "N/A"
        l_mean_V = l_std_V = l_median_V = l_iqr_V = "N/A"
        d_mean_Y = d_std_Y = d_median_Y = d_iqr_Y = "N/A"
        d_mean_H = d_std_H = d_median_H = d_iqr_H = "N/A"
        u_mean_Y = u_std_Y = u_median_Y = u_iqr_Y = "N/A"
        u_mean_H = u_std_H = u_median_H = u_iqr_H = "N/A"
        c_saccade_num = c_saccade_num_FR = "N/A"
        c_saccade_num_V = c_saccade_num_FR_V = "N/A"
        l_saccade_num = l_saccade_num_FR = "N/A"
        l_saccade_num_V = l_saccade_num_FR_V = "N/A"
        r_saccade_num = r_saccade_num_FR = "N/A"
        r_saccade_num_V = r_saccade_num_FR_V = "N/A"
        d_saccade_num = d_saccade_num_FR = "N/A"
        d_saccade_num_H = d_saccade_num_FR_H = "N/A"
        u_saccade_num = u_saccade_num_FR = "N/A"
        u_saccade_num_H = u_saccade_num_FR_H = "N/A"
        print("gaze_SPV occured ERROR.")
        logs = logs + "gaze_SPV occured ERROR." + "\n"

        return (c_mean_H, c_std_H, c_median_H, c_iqr_H,
                c_mean_V, c_std_V, c_median_V, c_iqr_V,
                r_mean_H, r_std_H, r_median_H, r_iqr_H,
                r_mean_V, r_std_V, r_median_V, r_iqr_V,
                l_mean_H, l_std_H, l_median_H, l_iqr_H,
                l_mean_V, l_std_V, l_median_V, l_iqr_V,
                d_mean_Y, d_std_Y, d_median_Y, d_iqr_Y,
                d_mean_H, d_std_H, d_median_H, d_iqr_H,
                u_mean_Y, u_std_Y, u_median_Y, u_iqr_Y,
                u_mean_H, u_std_H, u_median_H, u_iqr_H,
                c_saccade_num, c_saccade_num_FR,
                c_saccade_num_V, c_saccade_num_FR_V,
                l_saccade_num, l_saccade_num_FR,
                l_saccade_num_V, l_saccade_num_FR_V,
                r_saccade_num, r_saccade_num_FR,
                r_saccade_num_V, r_saccade_num_FR_V,
                d_saccade_num, d_saccade_num_FR,
                d_saccade_num_H, d_saccade_num_FR_H,
                u_saccade_num, u_saccade_num_FR,
                u_saccade_num_H, u_saccade_num_FR_H,
                logs)

# MAIN PROCESS

record = []
try:
    for i in range(0, len(lc.patient_test_data)):
        record.append({})

    for i in range(0, len(lc.patient_test_data)):

        ## get data
        per = "(" + str(i+1) + "/" + str(len(lc.patient_test_data)) + ") "
        print(per + "Now loading : " + lc.patient_test_data[i]['file_name'])
        logs = logs + per + "Now loading : " + lc.patient_test_data[i]['file_name'] + "\n"
        data_path = findout_csv_path(lc.patient_test_data[i]['file_name'])
        R_X, R_Y = righteye_angle_data(data_path)
        L_X, L_Y = lefteye_angle_data(data_path)

        #print(int(lc.patient_test_data[i]['point_x_position'][0]))

        for j in range(0, len(R_X)):
            R_X[j] += r_mod_x
            R_Y[j] += r_mod_y
            L_X[j] += l_mod_x
            L_Y[j] += l_mod_y

        R_X = np.asarray(R_X)
        L_X = np.asarray(L_X)
        R_Y = np.asarray(R_Y)
        L_Y = np.asarray(L_Y)

        ## modify the location of the target
        spot_x_angle = np.asarray(lc.patient_test_data[i]['point_x_position'])
        spot_y_angle = np.asarray(lc.patient_test_data[i]['point_y_position'])

        # pixel_to_angle (old ver.)
        #for j in range(0, len(lc.patient_test_data[i]['point_x_position'])):
        #    spot_x_angle[j] = x_radius_to_angle(int(lc.patient_test_data[i]['point_x_position'][j]))
        #    spot_y_angle[j] = y_radius_to_angle(int(lc.patient_test_data[i]['point_y_position'][j]))

        #print(lc.patient_test_data[i]['point_x_position'])
        #print(spot_x_angle, spot_y_angle)

        # Lefteye

        ## Calculate SPV

        data = pd.read_csv(data_path)

        Time = timeline(data.Lefteye_X)
        Time_E = Time

        # Left
        locs_H, pks_H, y_1_H, peaks_H, logs = Nystamus_extract(data.Lefteye_X, Fs, logs)
        locs_V, pks_V, y_1_V, peaks_V, logs = Nystamus_extract(data.Lefteye_Y, Fs, logs)

        # Right
        re_locs_H, re_pks_H, re_y_1_H, re_peaks_H, logs = Nystamus_extract(data.Righteye_X, Fs, logs)
        re_locs_V, re_pks_V, re_y_1_V, re_peaks_V, logs = Nystamus_extract(data.Righteye_Y, Fs, logs)

        # Left
        try:
            saccade_num_H = len(locs_H)
            saccade_num_V = len(locs_V)
            saccade_num_H_FR = round(saccade_num_H / Time[len(Time) - 1] * 10, 4)
            saccade_num_V_FR = round(saccade_num_V / Time[len(Time) - 1] * 10, 4)
        except:
            saccade_num_H = saccade_num_V = saccade_num_H_FR = saccade_num_V_FR = "N/A"

        # Right
        try:
            re_saccade_num_H = len(re_locs_H)
            re_saccade_num_V = len(re_locs_V)
            re_saccade_num_H_FR = round(re_saccade_num_H / Time[len(Time) - 1] * 10, 4)
            re_saccade_num_V_FR = round(re_saccade_num_V / Time[len(Time) - 1] * 10, 4)
        except:
            re_saccade_num_H = re_saccade_num_V = saccade_num_H_FR = saccade_num_V_FR = "N/A"

        # Left
        input_1_H, index_H, value_H, mean_H, std_H, median_H, iqr_H, logs = SPV_extract(data.Lefteye_X, Fs, logs)
        input_1_V, index_V, value_V, mean_V, std_V, median_V, iqr_V, logs = SPV_extract(data.Lefteye_Y, Fs, logs)

        # Right
        re_input_1_H, re_index_H, re_value_H, re_mean_H, re_std_H, re_median_H, re_iqr_H, logs = SPV_extract(data.Righteye_X, Fs, logs)
        re_input_1_V, re_index_V, re_value_V, re_mean_V, re_std_V, re_median_V, re_iqr_V, logs = SPV_extract(data.Righteye_Y, Fs, logs)

        ## modify the length of the data
        if len(Time) > len(spot_x_angle):
            Time = Time[0:len(spot_x_angle)]
            plt.plot(Time, spot_x_angle, 'salmon', label='Target_X', linewidth = 0.5, alpha = 0.5)
        else:
            spot_x_angle = spot_x_angle[0:len(Time)]
            plt.plot(Time, spot_x_angle, 'salmon', label='Target_X', linewidth = 0.5, alpha = 0.5)

        if len(Time) > len(spot_y_angle):
            Time = Time[0:len(spot_y_angle)]
            plt.plot(Time, spot_y_angle, 'royalblue', label='Target_Y', linewidth = 0.5, alpha = 0.5)
        else:
            spot_y_angle = spot_y_angle[0:len(Time)]
            plt.plot(Time, spot_y_angle, 'royalblue', label='Target_Y', linewidth = 0.5, alpha = 0.5)


        ## Specific test processing
        if 'Horizontal pursuit' in lc.patient_test_data[i]['file_name']:
            p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain, pp_med, pn_med, pm_med, logs = pursuit_gain(y_1_H, input_1_H, index_H, 'DEPRECATED', lc.patient_test_data[i]['Speed_Mode'], lc.patient_test_data[i]['Userdefined_speed'], lc.patient_test_data[i]['file_name'].split(' ')[0], logs)

            # Right-eye
            re_p_locs, re_n_locs, re_pg_locs, re_ng_locs, re_p_gain, re_n_gain, re_m_gain, re_pp_med, re_pn_med, re_pm_med, logs = pursuit_gain(re_y_1_H, re_input_1_H, re_index_H, 'DEPRECATED', lc.patient_test_data[i]['Speed_Mode'], lc.patient_test_data[i]['Userdefined_speed'], lc.patient_test_data[i]['file_name'].split(' ')[0], logs)

            #print(p_gain, n_gain, m_gain)

        if 'Vertical pursuit' in lc.patient_test_data[i]['file_name']:
            p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain, pp_med, pn_med, pm_med, logs = pursuit_gain(y_1_V, input_1_V, index_V, 'DEPRECATED', lc.patient_test_data[i]['Speed_Mode'], lc.patient_test_data[i]['Userdefined_speed'], lc.patient_test_data[i]['file_name'].split(' ')[0], logs)

            # Right-eye
            re_p_locs, re_n_locs, re_pg_locs, re_ng_locs, re_p_gain, re_n_gain, re_m_gain, re_pp_med, re_pn_med, re_pm_med, logs = pursuit_gain(re_y_1_V, re_input_1_V, re_index_V, 'DEPRECATED', lc.patient_test_data[i]['Speed_Mode'], lc.patient_test_data[i]['Userdefined_speed'], lc.patient_test_data[i]['file_name'].split(' ')[0], logs)

            #print(p_gain, n_gain, m_gain)


        if 'Horizontal saccade' in lc.patient_test_data[i]['file_name']:
            (p_locs, n_locs, p_first_amp_loc, n_first_amp_loc,
             r_gain, l_gain, m_gain, sr_med, sl_med, sm_med,
             p_peak_v, n_peak_v, m_peak_v, p_med_v, n_med_v, m_med_v, logs) = saccade_gain(y_1_H, input_1_H, locs_H, lc.patient_test_data[i]['Userdefined_speed'], logs)

            # Right-eye
            (re_p_locs, re_n_locs, re_p_first_amp_loc, re_n_first_amp_loc,
             re_r_gain, re_l_gain, re_m_gain, re_sr_med, re_sl_med, re_sm_med,
             re_p_peak_v, re_n_peak_v, re_m_peak_v, re_p_med_v, re_n_med_v, re_m_med_v, logs) = saccade_gain(re_y_1_H, re_input_1_H, re_locs_H, lc.patient_test_data[i]['Userdefined_speed'], logs)

            #print(r_gain, l_gain, p_peak_v, n_peak_v)

        if 'Vertical saccade' in lc.patient_test_data[i]['file_name']:
            (p_locs, n_locs, p_first_amp_loc, n_first_amp_loc,
             r_gain, l_gain, m_gain, sr_med, sl_med, sm_med,
             p_peak_v, n_peak_v, m_peak_v, p_med_v, n_med_v, m_med_v, logs) = saccade_gain(y_1_V, input_1_V, locs_V, lc.patient_test_data[i]['Userdefined_speed'], logs)

            # Right-eye
            (re_p_locs, re_n_locs, re_p_first_amp_loc, re_n_first_amp_loc,
             re_r_gain, re_l_gain, re_m_gain, re_sr_med, re_sl_med, re_sm_med,
             re_p_peak_v, re_n_peak_v, re_m_peak_v, re_p_med_v, re_n_med_v, re_m_med_v, logs) = saccade_gain(re_y_1_V, re_input_1_V, re_locs_V, lc.patient_test_data[i]['Userdefined_speed'], logs)

            #print(r_gain, l_gain, p_peak_v, n_peak_v)

        if 'Gaze' in lc.patient_test_data[i]['file_name']:
            (c_mean_H, c_std_H, c_median_H, c_iqr_H, c_mean_V, c_std_V, c_median_V, c_iqr_V,
             r_mean_H, r_std_H, r_median_H, r_iqr_H, r_mean_V, r_std_V, r_median_V, r_iqr_V,
             l_mean_H, l_std_H, l_median_H, l_iqr_H, l_mean_V, l_std_V, l_median_V, l_iqr_V,
             d_mean_Y, d_std_Y, d_median_Y, d_iqr_Y, d_mean_H, d_std_H, d_median_H, d_iqr_H,
             u_mean_Y, u_std_Y, u_median_Y, u_iqr_Y, u_mean_H, u_std_H, u_median_H, u_iqr_H,
             c_saccade_num, c_saccade_num_FR, c_saccade_num_V, c_saccade_num_FR_V,
             l_saccade_num, l_saccade_num_FR, l_saccade_num_V, l_saccade_num_FR_V,
             r_saccade_num, r_saccade_num_FR, r_saccade_num_V, r_saccade_num_FR_V,
             d_saccade_num, d_saccade_num_FR, d_saccade_num_H, d_saccade_num_FR_H,
             u_saccade_num, u_saccade_num_FR, u_saccade_num_H, u_saccade_num_FR_H,
             logs) = gaze_SPV(spot_x_angle, spot_y_angle, data.Lefteye_X, data.Lefteye_Y, logs)

            # Right-eye
            (re_c_mean_H, re_c_std_H, re_c_median_H, re_c_iqr_H, re_c_mean_V, re_c_std_V, re_c_median_V, re_c_iqr_V,
             re_r_mean_H, re_r_std_H, re_r_median_H, re_r_iqr_H, re_r_mean_V, re_r_std_V, re_r_median_V, re_r_iqr_V,
             re_l_mean_H, re_l_std_H, re_l_median_H, re_l_iqr_H, re_l_mean_V, re_l_std_V, re_l_median_V, re_l_iqr_V,
             re_d_mean_Y, re_d_std_Y, re_d_median_Y, re_d_iqr_Y, re_d_mean_H, re_d_std_H, re_d_median_H, re_d_iqr_H,
             re_u_mean_Y, re_u_std_Y, re_u_median_Y, re_u_iqr_Y, re_u_mean_H, re_u_std_H, re_u_median_H, re_u_iqr_H,
             re_c_saccade_num, re_c_saccade_num_FR, re_c_saccade_num_V, re_c_saccade_num_FR_V,
             re_l_saccade_num, re_l_saccade_num_FR, re_l_saccade_num_V, re_l_saccade_num_FR_V,
             re_r_saccade_num, re_r_saccade_num_FR, re_r_saccade_num_V, re_r_saccade_num_FR_V,
             re_d_saccade_num, re_d_saccade_num_FR, re_d_saccade_num_H, re_d_saccade_num_FR_H,
             re_u_saccade_num, re_u_saccade_num_FR, re_u_saccade_num_H, re_u_saccade_num_FR_H,
             logs) = gaze_SPV(spot_x_angle, spot_y_angle, data.Righteye_X, data.Righteye_Y, logs)

            #print(c_mean_H, u_mean_Y, c_saccade_num, c_saccade_num_FR)

        # if len(Time) > len(L_X):
        #     Time = Time[0:len(L_X)]
        # else:
        #     L_X = L_X[0:len(Time)]

        ## record data


        ### raw data

        record[i]['ID'] = patient_ID
        record[i]['file_name'] = lc.patient_test_data[i]['file_name']
        record[i]['R_X'] = re_y_1_H
        record[i]['L_X'] = y_1_H
        record[i]['R_Y'] = re_y_1_V
        record[i]['L_Y'] = y_1_V
        record[i]['spot_x_angle'] = spot_x_angle
        record[i]['spot_y_angle'] = spot_y_angle
        record[i]['Time'] = Time
        record[i]['Time_E'] = Time_E


        ### horizontal
        try:
            record[i]['l_mean_H'] = round_two(mean_H)
            record[i]['l_std_H'] = round_two(std_H)
            record[i]['l_median_H'] = round_two(median_H)
            record[i]['l_iqr_H'] = round_two(iqr_H)
            record[i]['l_max_H'] = round_two(max(value_H))
            record[i]['l_min_H'] = round_two(min(value_H))
            record[i]['l_sc_n'] = round_two(saccade_num_H)
            record[i]['l_FR'] = round_two(saccade_num_H_FR)
        except:
            record[i]['l_mean_H'] = mean_H
            record[i]['l_std_H'] = std_H
            record[i]['l_median_H'] = median_H
            record[i]['l_iqr_H'] = iqr_H
            record[i]['l_max_H'] = value_H
            record[i]['l_min_H'] = value_H
            record[i]['l_sc_n'] = saccade_num_H
            record[i]['l_FR'] = saccade_num_H_FR

        try:
            record[i]['r_mean_H'] = round_two(re_mean_H)
            record[i]['r_std_H'] = round_two(re_std_H)
            record[i]['r_median_H'] = round_two(re_median_H)
            record[i]['r_iqr_H'] = round_two(re_iqr_H)
            record[i]['r_max_H'] = round_two(max(re_value_H))
            record[i]['r_min_H'] = round_two(min(re_value_H))
            record[i]['r_sc_n'] = round_two(re_saccade_num_H)
            record[i]['r_FR'] = round_two(re_saccade_num_H_FR)
        except:
            record[i]['r_mean_H'] = re_mean_H
            record[i]['r_std_H'] = re_std_H
            record[i]['r_median_H'] = re_median_H
            record[i]['r_iqr_H'] = re_iqr_H
            record[i]['r_max_H'] = re_value_H
            record[i]['r_min_H'] = re_value_H
            record[i]['r_sc_n'] = re_saccade_num_H
            record[i]['r_FR'] = re_saccade_num_H_FR


        ## pursuit
        if 'pursuit' in lc.patient_test_data[i]['file_name']:
            try:
                record[i]['l_pp_gain'] = str(round_two(p_gain))
                record[i]['l_pn_gain'] = str(round_two(n_gain))
                record[i]['l_pm_gain'] = str(round_two(m_gain))
                record[i]['l_pp_med'] = str(round_two(pp_med))
                record[i]['l_pn_med'] = str(round_two(pn_med))
                record[i]['l_pm_med'] = str(round_two(pm_med))

            # N/A
            except:
                record[i]['l_pp_gain'] = p_gain
                record[i]['l_pn_gain'] = n_gain
                record[i]['l_pm_gain'] = m_gain
                record[i]['l_pp_med'] = pp_med
                record[i]['l_pn_med'] = pn_med
                record[i]['l_pm_med'] = pm_med

            # Right
            try:
                record[i]['r_pp_gain'] = str(round_two(re_p_gain))
                record[i]['r_pn_gain'] = str(round_two(re_n_gain))
                record[i]['r_pm_gain'] = str(round_two(re_m_gain))
                record[i]['r_pp_med'] = str(round_two(re_pp_med))
                record[i]['r_pn_med'] = str(round_two(re_pn_med))
                record[i]['r_pm_med'] = str(round_two(re_pm_med))
                # N/A
            except:
                record[i]['r_pp_gain'] = re_p_gain
                record[i]['r_pn_gain'] = re_n_gain
                record[i]['r_pm_gain'] = re_m_gain
                record[i]['r_pp_med'] = re_pp_med
                record[i]['r_pn_med'] = re_pn_med
                record[i]['r_pm_med'] = re_pm_med


        ## saccade

        if 'saccade' in lc.patient_test_data[i]['file_name']:
            try:
                record[i]['l_sp_gain'] = str(round_two(r_gain))
                record[i]['l_sn_gain'] = str(round_two(l_gain))
                record[i]['l_sm_gain'] = str(round_two(m_gain))
                record[i]['l_sp_med'] = str(round_two(sr_med))
                record[i]['l_sn_med'] = str(round_two(sl_med))
                record[i]['l_sm_med'] = str(round_two(sm_med))
            except:
                record[i]['l_sp_gain'] = r_gain
                record[i]['l_sn_gain'] = l_gain
                record[i]['l_sm_gain'] = m_gain
                record[i]['l_sp_med'] = sr_med
                record[i]['l_sn_med'] = sl_med
                record[i]['l_sm_med'] = sm_med

            try:
                record[i]['l_spv_p'] = str(int(p_peak_v))
                record[i]['l_spv_n'] = str(int(n_peak_v))
                record[i]['l_spv_m'] = str(int(m_peak_v))
                record[i]['l_spv_p_med'] = str(int(p_med_v))
                record[i]['l_spv_n_med'] = str(int(n_med_v))
                record[i]['l_spv_m_med'] = str(int(m_med_v))
            except:
                record[i]['l_spv_p'] = p_peak_v
                record[i]['l_spv_n'] = n_peak_v
                record[i]['l_spv_m'] = m_peak_v
                record[i]['l_spv_p_med'] = p_med_v
                record[i]['l_spv_n_med'] = n_med_v
                record[i]['l_spv_m_med'] = m_med_v

            try:
                record[i]['r_sp_gain'] = str(round_two(re_r_gain))
                record[i]['r_sn_gain'] = str(round_two(re_l_gain))
                record[i]['r_sm_gain'] = str(round_two(re_m_gain))
                record[i]['r_sp_med'] = str(round_two(re_sr_med))
                record[i]['r_sn_med'] = str(round_two(re_sl_med))
                record[i]['r_sm_med'] = str(round_two(re_sm_med))
            except:
                record[i]['r_sp_gain'] = re_r_gain
                record[i]['r_sn_gain'] = re_l_gain
                record[i]['r_sm_gain'] = re_m_gain
                record[i]['r_sp_med'] = re_sr_med
                record[i]['r_sn_med'] = re_sl_med
                record[i]['r_sm_med'] = re_sm_med

            try:
                record[i]['r_spv_p'] = str(int(re_p_peak_v))
                record[i]['r_spv_n'] = str(int(re_n_peak_v))
                record[i]['r_spv_m'] = str(int(re_m_peak_v))
                record[i]['r_spv_p_med'] = str(int(re_p_med_v))
                record[i]['r_spv_n_med'] = str(int(re_n_med_v))
                record[i]['r_spv_m_med'] = str(int(re_m_med_v))
            except:
                record[i]['r_spv_p'] = re_p_peak_v
                record[i]['r_spv_n'] = re_n_peak_v
                record[i]['r_spv_m'] = re_m_peak_v
                record[i]['r_spv_p_med'] = re_p_med_v
                record[i]['r_spv_n_med'] = re_n_med_v
                record[i]['r_spv_m_med'] = re_m_med_v


        ## gaze evoked

        if 'Gaze' in lc.patient_test_data[i]['file_name']:
            ### Center
            try: # OS_Hor.
                record[i]['l_ge_c_mean'] = round_two(c_mean_H)
                record[i]['l_ge_c_std'] = round_two(c_std_H)
                record[i]['l_ge_c_median'] = round_two(c_median_H)
                record[i]['l_ge_c_iqr'] = round_two(c_iqr_H)
                record[i]['l_ge_c_st'] = round_two(c_saccade_num)
                record[i]['l_ge_c_fr'] = round_two(c_saccade_num_FR)
            except:
                record[i]['l_ge_c_mean'] = c_mean_H
                record[i]['l_ge_c_std'] = c_std_H
                record[i]['l_ge_c_median'] = c_median_H
                record[i]['l_ge_c_iqr'] = c_iqr_H
                record[i]['l_ge_c_st'] = c_saccade_num
                record[i]['l_ge_c_fr'] = c_saccade_num_FR

            try: # OS_Ver.
                record[i]['l_ge_c_mean_V'] = round_two(c_mean_V)
                record[i]['l_ge_c_std_V'] = round_two(c_std_V)
                record[i]['l_ge_c_median_V'] = round_two(c_median_V)
                record[i]['l_ge_c_iqr_V'] = round_two(c_iqr_V)
                record[i]['l_ge_c_st_V'] = round_two(c_saccade_num_V)
                record[i]['l_ge_c_fr_V'] = round_two(c_saccade_num_FR_V)
            except:
                record[i]['l_ge_c_mean_V'] = c_mean_V
                record[i]['l_ge_c_std_V'] = c_std_V
                record[i]['l_ge_c_median_V'] = c_median_V
                record[i]['l_ge_c_iqr_V'] = c_iqr_V
                record[i]['l_ge_c_st_V'] = c_saccade_num_V
                record[i]['l_ge_c_fr_V'] = c_saccade_num_FR_V

            try: # OD_Hor
                record[i]['r_ge_c_mean'] = round_two(re_c_mean_H)
                record[i]['r_ge_c_std'] = round_two(re_c_std_H)
                record[i]['r_ge_c_median'] = round_two(re_c_median_H)
                record[i]['r_ge_c_iqr'] = round_two(re_c_iqr_H)
                record[i]['r_ge_c_st'] = round_two(re_c_saccade_num)
                record[i]['r_ge_c_fr'] = round_two(re_c_saccade_num_FR)
            except:
                record[i]['r_ge_c_mean'] = re_c_mean_H
                record[i]['r_ge_c_std'] = re_c_std_H
                record[i]['r_ge_c_median'] = re_c_median_H
                record[i]['r_ge_c_iqr'] = re_c_iqr_H
                record[i]['r_ge_c_st'] = re_c_saccade_num
                record[i]['r_ge_c_fr'] = re_c_saccade_num_FR

            try: # OD_VER.
                record[i]['r_ge_c_mean_V'] = round_two(re_c_mean_V)
                record[i]['r_ge_c_std_V'] = round_two(re_c_std_V)
                record[i]['r_ge_c_median_V'] = round_two(re_c_median_V)
                record[i]['r_ge_c_iqr_V'] = round_two(re_c_iqr_V)
                record[i]['r_ge_c_st_V'] = round_two(re_c_saccade_num_V)
                record[i]['r_ge_c_fr_V'] = round_two(re_c_saccade_num_FR_V)

            except:
                record[i]['r_ge_c_mean_V'] = re_c_mean_V
                record[i]['r_ge_c_std_V'] = re_c_std_V
                record[i]['r_ge_c_median_V'] = re_c_median_V
                record[i]['r_ge_c_iqr_V'] = re_c_iqr_V
                record[i]['r_ge_c_st_V'] = re_c_saccade_num_V
                record[i]['r_ge_c_fr_V'] = re_c_saccade_num_FR_V

            ### RIGHT
            try: # OS
                record[i]['l_ge_r_mean'] = round_two(r_mean_H)
                record[i]['l_ge_r_std'] = round_two(r_std_H)
                record[i]['l_ge_r_median'] = round_two(r_median_H)
                record[i]['l_ge_r_iqr'] = round_two(r_iqr_H)
                record[i]['l_ge_r_st'] = round_two(r_saccade_num)
                record[i]['l_ge_r_fr'] = round_two(r_saccade_num_FR)
            except:
                record[i]['l_ge_r_mean'] = r_mean_H
                record[i]['l_ge_r_std'] = r_std_H
                record[i]['l_ge_r_median'] = r_median_H
                record[i]['l_ge_r_iqr'] = r_iqr_H
                record[i]['l_ge_r_st'] = r_saccade_num
                record[i]['l_ge_r_fr'] = r_saccade_num_FR

            try:
                record[i]['l_ge_r_mean_V'] = round_two(r_mean_V)
                record[i]['l_ge_r_std_V'] = round_two(r_std_V)
                record[i]['l_ge_r_median_V'] = round_two(r_median_V)
                record[i]['l_ge_r_iqr_V'] = round_two(r_iqr_V)
                record[i]['l_ge_r_st_V'] = round_two(r_saccade_num_V)
                record[i]['l_ge_r_fr_V'] = round_two(r_saccade_num_FR_V)
            except:
                record[i]['l_ge_r_mean_V'] = r_mean_V
                record[i]['l_ge_r_std_V'] = r_std_V
                record[i]['l_ge_r_median_V'] = r_median_V
                record[i]['l_ge_r_iqr_V'] = r_iqr_V
                record[i]['l_ge_r_st_V'] = r_saccade_num_V
                record[i]['l_ge_r_fr_V'] = r_saccade_num_FR_V

            try:
                # OD
                record[i]['r_ge_r_mean'] = round_two(re_r_mean_H)
                record[i]['r_ge_r_std'] = round_two(re_r_std_H)
                record[i]['r_ge_r_median'] = round_two(re_r_median_H)
                record[i]['r_ge_r_iqr'] = round_two(re_r_iqr_H)
                record[i]['r_ge_r_st'] = round_two(re_r_saccade_num)
                record[i]['r_ge_r_fr'] = round_two(re_r_saccade_num_FR)
            except:
                record[i]['r_ge_r_mean'] = re_r_mean_H
                record[i]['r_ge_r_std'] = re_r_std_H
                record[i]['r_ge_r_median'] = re_r_median_H
                record[i]['r_ge_r_iqr'] = re_r_iqr_H
                record[i]['r_ge_r_st'] = re_r_saccade_num
                record[i]['r_ge_r_fr'] = re_r_saccade_num_FR

            try:
                record[i]['r_ge_r_mean_V'] = round_two(re_r_mean_V)
                record[i]['r_ge_r_std_V'] = round_two(re_r_std_V)
                record[i]['r_ge_r_median_V'] = round_two(re_r_median_V)
                record[i]['r_ge_r_iqr_V'] = round_two(re_r_iqr_V)
                record[i]['r_ge_r_st_V'] = round_two(re_r_saccade_num_V)
                record[i]['r_ge_r_fr_V'] = round_two(re_r_saccade_num_FR_V)
            except:
                record[i]['r_ge_r_mean_V'] = re_r_mean_V
                record[i]['r_ge_r_std_V'] = re_r_std_V
                record[i]['r_ge_r_median_V'] = re_r_median_V
                record[i]['r_ge_r_iqr_V'] = re_r_iqr_V
                record[i]['r_ge_r_st_V'] = re_r_saccade_num_V
                record[i]['r_ge_r_fr_V'] = re_r_saccade_num_FR_V

            ### LEFT
            try: # OS
                record[i]['l_ge_l_mean'] = round_two(l_mean_H)
                record[i]['l_ge_l_std'] = round_two(l_std_H)
                record[i]['l_ge_l_median'] = round_two(l_median_H)
                record[i]['l_ge_l_iqr'] = round_two(l_iqr_H)
                record[i]['l_ge_l_st'] = round_two(l_saccade_num)
                record[i]['l_ge_l_fr'] = round_two(l_saccade_num_FR)
            except:
                record[i]['l_ge_l_mean'] = l_mean_H
                record[i]['l_ge_l_std'] = l_std_H
                record[i]['l_ge_l_median'] = l_median_H
                record[i]['l_ge_l_iqr'] = l_iqr_H
                record[i]['l_ge_l_st'] = l_saccade_num
                record[i]['l_ge_l_fr'] = l_saccade_num_FR

            try:
                record[i]['l_ge_l_mean_V'] = round_two(l_mean_V)
                record[i]['l_ge_l_std_V'] = round_two(l_std_V)
                record[i]['l_ge_l_median_V'] = round_two(l_median_V)
                record[i]['l_ge_l_iqr_V'] = round_two(l_iqr_V)
                record[i]['l_ge_l_st_V'] = round_two(l_saccade_num_V)
                record[i]['l_ge_l_fr_V'] = round_two(l_saccade_num_FR_V)
            except:
                record[i]['l_ge_l_mean_V'] = l_mean_V
                record[i]['l_ge_l_std_V'] = l_std_V
                record[i]['l_ge_l_median_V'] = l_median_V
                record[i]['l_ge_l_iqr_V'] = l_iqr_V
                record[i]['l_ge_l_st_V'] = l_saccade_num_V
                record[i]['l_ge_l_fr_V'] = l_saccade_num_FR_V

            try: # OD
                record[i]['r_ge_l_mean'] = round_two(re_l_mean_H)
                record[i]['r_ge_l_std'] = round_two(re_l_std_H)
                record[i]['r_ge_l_median'] = round_two(re_l_median_H)
                record[i]['r_ge_l_iqr'] = round_two(re_l_iqr_H)
                record[i]['r_ge_l_st'] = round_two(re_l_saccade_num)
                record[i]['r_ge_l_fr'] = round_two(re_l_saccade_num_FR)
            except:
                record[i]['r_ge_l_mean'] = re_l_mean_H
                record[i]['r_ge_l_std'] = re_l_std_H
                record[i]['r_ge_l_median'] = re_l_median_H
                record[i]['r_ge_l_iqr'] = re_l_iqr_H
                record[i]['r_ge_l_st'] = re_l_saccade_num
                record[i]['r_ge_l_fr'] = re_l_saccade_num_FR

            try:
                record[i]['r_ge_l_mean_V'] = round_two(re_l_mean_V)
                record[i]['r_ge_l_std_V'] = round_two(re_l_std_V)
                record[i]['r_ge_l_median_V'] = round_two(re_l_median_V)
                record[i]['r_ge_l_iqr_V'] = round_two(re_l_iqr_V)
                record[i]['r_ge_l_st_V'] = round_two(re_l_saccade_num_V)
                record[i]['r_ge_l_fr_V'] = round_two(re_l_saccade_num_FR_V)
            except:
                record[i]['r_ge_l_mean_V'] = re_l_mean_V
                record[i]['r_ge_l_std_V'] = re_l_std_V
                record[i]['r_ge_l_median_V'] = re_l_median_V
                record[i]['r_ge_l_iqr_V'] = re_l_iqr_V
                record[i]['r_ge_l_st_V'] = re_l_saccade_num_V
                record[i]['r_ge_l_fr_V'] = re_l_saccade_num_FR_V

            # DOWN
            try: # OS
                record[i]['l_ge_d_mean'] = round_two(d_mean_Y)
                record[i]['l_ge_d_std'] = round_two(d_std_Y)
                record[i]['l_ge_d_median'] = round_two(d_median_Y)
                record[i]['l_ge_d_iqr'] = round_two(d_iqr_Y)
                record[i]['l_ge_d_st'] = round_two(d_saccade_num)
                record[i]['l_ge_d_fr'] = round_two(d_saccade_num_FR)
            except:
                record[i]['l_ge_d_mean'] = d_mean_Y
                record[i]['l_ge_d_std'] = d_std_Y
                record[i]['l_ge_d_median'] = d_median_Y
                record[i]['l_ge_d_iqr'] = d_iqr_Y
                record[i]['l_ge_d_st'] = d_saccade_num
                record[i]['l_ge_d_fr'] = d_saccade_num_FR

            try:
                record[i]['l_ge_d_mean_H'] = round_two(d_mean_H)
                record[i]['l_ge_d_std_H'] = round_two(d_std_H)
                record[i]['l_ge_d_median_H'] = round_two(d_median_H)
                record[i]['l_ge_d_iqr_H'] = round_two(d_iqr_H)
                record[i]['l_ge_d_st_H'] = round_two(d_saccade_num_H)
                record[i]['l_ge_d_fr_H'] = round_two(d_saccade_num_FR_H)
            except:
                record[i]['l_ge_d_mean_H'] = d_mean_H
                record[i]['l_ge_d_std_H'] = d_std_H
                record[i]['l_ge_d_median_H'] = d_median_H
                record[i]['l_ge_d_iqr_H'] = d_iqr_H
                record[i]['l_ge_d_st_H'] = d_saccade_num_H
                record[i]['l_ge_d_fr_H'] = d_saccade_num_FR_H

            try: # OD
                record[i]['r_ge_d_mean'] = round_two(re_d_mean_Y)
                record[i]['r_ge_d_std'] = round_two(re_d_std_Y)
                record[i]['r_ge_d_median'] = round_two(re_d_median_Y)
                record[i]['r_ge_d_iqr'] = round_two(re_d_iqr_Y)
                record[i]['r_ge_d_st'] = round_two(re_d_saccade_num)
                record[i]['r_ge_d_fr'] = round_two(re_d_saccade_num_FR)

            except:
                record[i]['r_ge_d_mean'] = re_d_mean_Y
                record[i]['r_ge_d_std'] = re_d_std_Y
                record[i]['r_ge_d_median'] = re_d_median_Y
                record[i]['r_ge_d_iqr'] = re_d_iqr_Y
                record[i]['r_ge_d_st'] = re_d_saccade_num
                record[i]['r_ge_d_fr'] = re_d_saccade_num_FR

            try:
                record[i]['r_ge_d_mean_H'] = round_two(re_d_mean_H)
                record[i]['r_ge_d_std_H'] = round_two(re_d_std_H)
                record[i]['r_ge_d_median_H'] = round_two(re_d_median_H)
                record[i]['r_ge_d_iqr_H'] = round_two(re_d_iqr_H)
                record[i]['r_ge_d_st_H'] = round_two(re_d_saccade_num_H)
                record[i]['r_ge_d_fr_H'] = round_two(re_d_saccade_num_FR_H)
            except:
                record[i]['r_ge_d_mean_H'] = re_d_mean_H
                record[i]['r_ge_d_std_H'] = re_d_std_H
                record[i]['r_ge_d_median_H'] = re_d_median_H
                record[i]['r_ge_d_iqr_H'] = re_d_iqr_H
                record[i]['r_ge_d_st_H'] = re_d_saccade_num_H
                record[i]['r_ge_d_fr_H'] = re_d_saccade_num_FR_H

            ### UP
            try: # OS
                record[i]['l_ge_u_mean'] = round_two(u_mean_Y)
                record[i]['l_ge_u_std'] = round_two(u_std_Y)
                record[i]['l_ge_u_median'] = round_two(u_median_Y)
                record[i]['l_ge_u_iqr'] = round_two(u_iqr_Y)
                record[i]['l_ge_u_st'] = round_two(u_saccade_num)
                record[i]['l_ge_u_fr'] = round_two(u_saccade_num_FR)
            except:
                record[i]['l_ge_u_mean'] = u_mean_Y
                record[i]['l_ge_u_std'] = u_std_Y
                record[i]['l_ge_u_median'] = u_median_Y
                record[i]['l_ge_u_iqr'] = u_iqr_Y
                record[i]['l_ge_u_st'] = u_saccade_num
                record[i]['l_ge_u_fr'] = u_saccade_num_FR

            try:
                record[i]['l_ge_u_mean_H'] = round_two(u_mean_H)
                record[i]['l_ge_u_std_H'] = round_two(u_std_H)
                record[i]['l_ge_u_median_H'] = round_two(u_median_H)
                record[i]['l_ge_u_iqr_H'] = round_two(u_iqr_H)
                record[i]['l_ge_u_st_H'] = round_two(u_saccade_num_H)
                record[i]['l_ge_u_fr_H'] = round_two(u_saccade_num_FR_H)
            except:
                record[i]['l_ge_u_mean_H'] = u_mean_H
                record[i]['l_ge_u_std_H'] = u_std_H
                record[i]['l_ge_u_median_H'] = u_median_H
                record[i]['l_ge_u_iqr_H'] = u_iqr_H
                record[i]['l_ge_u_st_H'] = u_saccade_num_H
                record[i]['l_ge_u_fr_H'] = u_saccade_num_FR_H

            try: # OD
                record[i]['r_ge_u_mean'] = round_two(re_u_mean_Y)
                record[i]['r_ge_u_std'] = round_two(re_u_std_Y)
                record[i]['r_ge_u_median'] = round_two(re_u_median_Y)
                record[i]['r_ge_u_iqr'] = round_two(re_u_iqr_Y)
                record[i]['r_ge_u_st'] = round_two(re_u_saccade_num)
                record[i]['r_ge_u_fr'] = round_two(re_u_saccade_num_FR)
            except:
                record[i]['r_ge_u_mean'] = re_u_mean_Y
                record[i]['r_ge_u_std'] = re_u_std_Y
                record[i]['r_ge_u_median'] = re_u_median_Y
                record[i]['r_ge_u_iqr'] = re_u_iqr_Y
                record[i]['r_ge_u_st'] = re_u_saccade_num
                record[i]['r_ge_u_fr'] = re_u_saccade_num_FR

            try:
                record[i]['r_ge_u_mean_H'] = round_two(re_u_mean_H)
                record[i]['r_ge_u_std_H'] = round_two(re_u_std_H)
                record[i]['r_ge_u_median_H'] = round_two(re_u_median_H)
                record[i]['r_ge_u_iqr_H'] = round_two(re_u_iqr_H)
                record[i]['r_ge_u_st_H'] = round_two(re_u_saccade_num_H)
                record[i]['r_ge_u_fr_H'] = round_two(re_u_saccade_num_FR_H)
            except:
                record[i]['r_ge_u_mean_H'] = re_u_mean_H
                record[i]['r_ge_u_std_H'] = re_u_std_H
                record[i]['r_ge_u_median_H'] = re_u_median_H
                record[i]['r_ge_u_iqr_H'] = re_u_iqr_H
                record[i]['r_ge_u_st_H'] = re_u_saccade_num_H
                record[i]['r_ge_u_fr_H'] = re_u_saccade_num_FR_H

        ### vertical
        try:
            record[i]['l_mean_V'] = round_two(mean_V)
            record[i]['l_std_V'] = round_two(std_V)
            record[i]['l_median_V'] = round_two(median_V)
            record[i]['l_iqr_V'] = round_two(iqr_V)
            record[i]['l_max_V'] = round_two(max(value_V))
            record[i]['l_min_V'] = round_two(min(value_V))
            record[i]['l_sc_n_V'] = round_two(saccade_num_V)
            record[i]['l_FR_V'] = round_two(saccade_num_V_FR)
        except:
            record[i]['l_mean_V'] = mean_V
            record[i]['l_std_V'] = std_V
            record[i]['l_median_V'] = median_V
            record[i]['l_iqr_V'] = iqr_V
            record[i]['l_max_V'] = value_V
            record[i]['l_min_V'] = value_V
            record[i]['l_sc_n_V'] = saccade_num_V
            record[i]['l_FR_V'] = saccade_num_V_FR

        try:
            record[i]['r_mean_V'] = round_two(re_mean_V)
            record[i]['r_std_V'] = round_two(re_std_V)
            record[i]['r_median_V'] = round_two(re_median_V)
            record[i]['r_iqr_V'] = round_two(re_iqr_V)
            record[i]['r_max_V'] = round_two(max(re_value_V))
            record[i]['r_min_V'] = round_two(min(re_value_V))
            record[i]['r_sc_n_V'] = round_two(re_saccade_num_V)
            record[i]['r_FR_V'] = round_two(re_saccade_num_V_FR)
        except:
            record[i]['r_mean_V'] = re_mean_V
            record[i]['r_std_V'] = re_std_V
            record[i]['r_median_V'] = re_median_V
            record[i]['r_iqr_V'] = re_iqr_V
            record[i]['r_max_V'] = re_value_V
            record[i]['r_min_V'] = re_value_V
            record[i]['r_sc_n_V'] = re_saccade_num_V
            record[i]['r_FR_V'] = re_saccade_num_V_FR

        ## plot
        # plt.plot(Time, L_X, 'r', label='Patient', linewidth = 0.5)
        try:
            if "saccade" in lc.patient_test_data[i]['file_name']:
                plt.plot(p_locs / 25, y_1_H[re_p_locs], 'g.', label='p_locs')
                plt.plot(n_locs / 25, y_1_H[re_n_locs], 'm.', label='n_locs')
                plt.plot(p_first_amp_loc / 25, y_1_H[p_first_amp_loc], 'c.', label='pf_locs')
                plt.plot(n_first_amp_loc / 25, y_1_H[n_first_amp_loc], 'k.', label='nf_locs')
            plt.plot(Time_E, y_1_H, 'r', label='Patient_X', linewidth=0.5)
            plt.plot(Time_E, y_1_V, 'b', label='Patient_Y', linewidth=0.5)
            plt.plot(locs_H / 25, y_1_H[locs_H], 'b.', label='Saccade Trial', ms = 1)
            plt.plot(index_H / 25, y_1_H[index_H], 'y.', label='SPV', ms = 1)

            # if 'OKN' in lc.patient_test_data[i]['file_name']:
            #     plt.plot(index_H / 25, y_1_H[index_H],'g.', label='SPV', ms=1)

            # check algo.
            # plt.plot(Time, y_2_H, 'y', label='y_2', linewidth=0.5)
            # plt.plot(Time, y_3_H, 'lightcoral', label='y_3', linewidth=0.5)
            # plt.plot(Time, y_4_H, 'indianred', label='y_4', linewidth=0.5)
            # plt.plot(Time, y_5_H, 'brown', label='y_5', linewidth=0.5)
            # lt.plot(Time, y_6_H, 'firebrick', label='y_6', linewidth=0.5)
            # plt.plot(Time[:-1], y_7_H , 'b', label='y_7', linewidth = 0.5)
            # plt.plot(peaks_H/25, y_7_H[peaks_H], 'r.' , label='y_7_peaks', ms=1)

            plt.xlabel("Time(s)")
            plt.ylabel("Eye position(°)")
            plt.title(lc.patient_test_data[i]['file_name'])
            plt.legend(loc='upper right')
            plt.ylim(-30, 30)

            img_name = "Left_eye_" + lc.patient_test_data[i]['file_name'] + ".png"
            print("Createing : " + img_name)
            logs = logs + "Createing : " + img_name + "\n"
            plt.savefig(img_name, dpi=300)
            plt.close()

        except:
            print("LeftEye Ploting occured ERROR")
            logs = logs + "LeftEye Ploting occured ERROR" + "\n"

        # print(record[i]['L_X'])
        try:
            if "saccade" in lc.patient_test_data[i]['file_name']:
                plt.plot(re_p_locs / 25, re_y_1_H[re_p_locs], 'g.', label='p_locs')
                plt.plot(re_n_locs / 25, re_y_1_H[re_n_locs], 'm.', label='n_locs')
                plt.plot(re_p_first_amp_loc / 25, re_y_1_H[re_p_first_amp_loc], 'c.', label='pf_locs')
                plt.plot(re_n_first_amp_loc / 25, re_y_1_H[re_n_first_amp_loc], 'k.', label='nf_locs')
            plt.plot(Time_E, re_y_1_H, 'r', label='Patient_X', linewidth=0.5)
            plt.plot(Time_E, re_y_1_V, 'b', label='Patient_Y', linewidth=0.5)
            plt.plot(re_locs_H / 25, re_y_1_H[re_locs_H], 'b.', label='Saccade Trial', ms=1)
            plt.plot(re_index_H / 25, re_y_1_H[re_index_H], 'y.', label='SPV', ms=1)
            plt.plot(Time, spot_x_angle, 'salmon', label='Target_X', linewidth=0.5, alpha = 0.5)
            plt.plot(Time, spot_y_angle, 'royalblue', label='Target_Y', linewidth=0.5, alpha = 0.5)
            plt.xlabel("Time(s)")
            plt.ylabel("Eye position(°)")
            plt.title(lc.patient_test_data[i]['file_name'])
            plt.legend(loc='upper right')
            plt.ylim(-30, 30)

            img_name = "Right_eye_" + lc.patient_test_data[i]['file_name'] + ".png"
            print("Createing: " + img_name)
            logs = logs + "Createing : " + img_name + "\n"
            plt.savefig(img_name, dpi=300)

            # plt.show()
            plt.close()
        except:
            print("RighttEye Ploting occured ERROR")
            logs = logs + "RightEye Ploting occured ERROR" + "\n"

        try:
            print("Lefteye_H: mean =  %f | std = %f | median = %f | iqr = %f" % (mean_H,std_H,median_H,iqr_H))
            print("Lefteye_V: mean =  %f | std = %f | median = %f | iqr = %f" % (mean_V, std_V, median_V, iqr_V))
            logs = logs + "Lefteye_H: mean =  %f | std = %f | median = %f | iqr = %f" % (mean_H,std_H,median_H,iqr_H) + "\n"
            logs = logs + "Lefteye_V: mean =  %f | std = %f | median = %f | iqr = %f" % (mean_V, std_V, median_V, iqr_V) + "\n"
        except:
            print("LeftEye SPV Calculation occurd ERROR.")
            logs = logs + "LeftEye SPV Calculation occurd ERROR." + "\n"

        try:
            print("Righteye_H: mean =  %f | std = %f | median = %f | iqr = %f" % (re_mean_H, re_std_H, re_median_H, iqr_H))
            print("Righteye_V: mean =  %f | std = %f | median = %f | iqr = %f" % (re_mean_V, re_std_V, re_median_V, iqr_V))
            logs = logs + "Righteye_H: mean =  %f | std = %f | median = %f | iqr = %f" % (re_mean_H, re_std_H, re_median_H, iqr_H) + "\n"
            logs = logs + "Righteye_V: mean =  %f | std = %f | median = %f | iqr = %f" % (re_mean_V, re_std_V, re_median_V, iqr_V) + "\n"
        except:
            print("RightEye SPV Calculation occurd ERROR.")
            logs = logs + "RightEye SPV Calculation occurd ERROR." + "\n"

        print("Finish :" + lc.patient_test_data[i]['file_name'])
        print("===============================================")

        logs = logs + "Finish :" + lc.patient_test_data[i]['file_name'] + "\n"
        logs = logs + "===============================================" + "\n"
except:
    e_logs = traceback.format_exc()
    print("SPV_main.py: " + e_logs)
    logs_name = os.path.join(lc.error_path, "SPV_main.txt")
    fp = open(logs_name, "w")
    fp.write(e_logs)
    fp.close
    print("SPV_main.py occured ERROR.")

    logs = logs + "SPV_main.py: " + e_logs + "\n"
    logs = logs + "SPV_main.py occured ERROR." + "\n"


# LOGS
logs_N = os.path.join(lc.error_path, "LOGS.txt")
lp = open(logs_N, "a")
lp.write(logs)
lp.close
