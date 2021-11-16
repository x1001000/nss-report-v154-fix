import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import lfilter
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import math

def timeline(length):
    Time = []
    for j in range(0, len(length)):
        Time.append(round(j/Fs,2))
    return Time

def SPV_extract(input, Fs):
    # define SPV feature
    interval = 1 / Fs
    # print(input)
    input = signal.medfilt(input, 11)
    # print(input)
    y_1 = input - np.mean(input)
    # find out SPV index
    input_1 = np.diff(y_1) / interval
    zero_idx = np.where(input_1 == 0)
    input_nonzero = np.delete(input_1, zero_idx)

    c = 1.4826  # c=-1/(sqrt(2)*erfcinv(3/2))
    MAD = c * np.median(abs(input_nonzero - np.median(input_nonzero)))  # MAD = c*median(abs(A-median(A)))
    input_2 = (abs(input_nonzero - np.median(input_nonzero)) / MAD) > 2
    input_idx = input_2

    # find SPV index as 0 and 1
    for i in range(1, (len(input_2) - 2)):
        if (input_2[i - 1] & input_2[i + 1]) == 1: # F & F == F
            input_2[i] = 1
        elif (input_2[i - 1] | input_2[i + 1]) == 0:
            input_2[i] = 0
        else:
            input_2[i] = input_2[i]
    input_3 = ~input_2

    # idx = 0
    # for i in range(len(input_3)):
    #     if i == zero_idx[0][idx]:
    #         input_3[i] = False
    #         if idx < len(zero_idx[0]) - 1:
    #             idx += 1
    #         else:
    #             break

    # find SPV index
    for i in range(0, len(zero_idx[0])):
        input_3 = np.insert(input_3, zero_idx[0][i], None)

    tmp = np.where(input_3 == 1)
    index = tmp[0]
    y_2 = signal.medfilt(input_nonzero, 11)
    y_idx = y_2
    for i in range(0, len(zero_idx[0])):
        y_idx = np.insert(y_idx, zero_idx[0][i], 0)
    y_3 = []
    out = []
    for i in range(len(index)):
        y_3 = np.append(y_3, y_idx[index[i]])
        out = np.append(out, input_1[index[i]])

    #y_4 = [x for x in y_3 if (x > np.median(y_3) - 2 * np.std(y_3)) & (x < np.median(y_3) + 2 * np.std(y_3))]
    y_4 = y_3

    # compute SPV mean std value
    mean = np.mean(y_4)
    std = np.std(y_4)
    median = np.median(y_4)
    q1, q3 = np.percentile(y_4, [25, 75])
    iqr = q3 - q1
    # iqr = 0
    return input_1, index, y_3, y_2, y_1, out, input_2, input_3, mean, std, median, iqr

def modified_z_score(ys):
    c1 = 1.253  # When MAD is equal to ZERO.
    c2 = 1.486  # When MAD isn't equal to ZERO.
    median_y = np.median(ys)
    MAD_y = c2 * np.median(abs(ys - median_y))

    modified_z_score = [(y - median_y) / MAD_y for y in ys]

    return modified_z_score

def Nystamus_extract(y, Fs):
    ## Preprocessing stage
    y = signal.medfilt(y, 11)
    y_1 = y - np.mean(y)
    print(y_1)
    y_2 = modified_z_score(y_1)  # normalized amplitude
    print(y_2)
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

    y_f = []
    for j in range(0, len(locs_f)):
         y_f.append(y[locs_f[j]])

    y_2_f = []
    for j in range(0, len(locs_f)):
        y_2_f.append(y_2[locs_f[j]])

    y_7_f = []
    for j in range(0, len(locs_f)):
        y_7_f.append(y_7[locs_f[j]])

    return locs, pks, y_f, y_1


def pursuit_gain(input, input_v, SPV_idx, speed):
    target_velocity_3s = 20  # (15° - (- 15°)) / 1.5 = 20 °/s (Pursuit 3s)
    target_velocity_5s = 12  # (15° - (- 15°)) / 2.5 = 12 °/s (Pursuit 5s)
    target_velocity_10s = 6  # (15° - (- 15°)) / 5 = 6 °/s (Pursuit 10s)

    SPV = input_v[SPV_idx]

    p_locs = np.asarray(find_peaks(input, height=5, distance=speed / 2 * 25)[0])
    n_locs = np.asarray(find_peaks(-input, height=5, distance=speed / 2 * 25)[0])

    m_locs = np.sort(np.concatenate((p_locs, n_locs)))

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

    if speed == 3:
        p_gain = abs(np.mean(p_v)) / target_velocity_3s
        n_gain = abs(np.mean(n_v)) / target_velocity_3s
        m_gain = (p_gain + n_gain) / 2
    elif speed == 5:
        p_gain = abs(np.median(p_v)) / target_velocity_5s
        n_gain = abs(np.median(n_v)) / target_velocity_5s
        m_gain = (p_gain + n_gain) / 2
    elif speed == 10:
        p_gain = abs(np.mean(p_v)) / target_velocity_10s
        n_gain = abs(np.mean(n_v)) / target_velocity_10s
        m_gain = (p_gain + n_gain) / 2

    print(p_gain, n_gain, m_gain)

    return p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain

def saccade_gain(input, input_v, FP_idx, speed):
    p_locs = find_peaks(input, height=5, distance=speed / 2 * 25)
    n_locs = find_peaks(-input, height=5, distance=speed / 2 * 25)

    p_pks = []
    n_pks = []
    for i in range(len(p_locs[0])):
        p_pks.append(input[p_locs[0][i]])

    for i in range(len(n_locs[0])):
        n_pks.append(input[n_locs[0][i]])

    p_pks = np.asarray(p_pks)
    n_pks = np.asarray(n_pks)

    p_first_amp_loc = []
    p_gain_amp_loc = []
    n_first_amp_loc = []
    n_gain_amp_loc = []
    p_f_amp = []
    n_f_amp = []

    for i in range(0, len(FP_idx)):
        if input[FP_idx[i]] < 0:
            p_gain_amp_loc.append(FP_idx[i])
            p_first_amp_loc.append(FP_idx[i]+1)
        elif input[FP_idx[i]] > 0:
            n_gain_amp_loc.append(FP_idx[i])
            n_first_amp_loc.append(FP_idx[i]+1)

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

    print(r_gain, l_gain, m_gain, p_peak_v, n_peak_v, m_peak_v)

    return p_locs[0], n_locs[0], p_first_amp_loc, n_first_amp_loc, p_gain_amp_loc, n_gain_amp_loc, r_gain, l_gain, p_peak_v, n_peak_v


def gaze_SPV(target_H, target_V, input):
    center = []
    right = []
    left = []
    down = []
    up = []

    return 0


data = pd.read_csv("/Users/peterliang/Desktop/Neruobit_Python/Result/20200527 B100533006/Report/Horizontal saccade 5s/Horizontal saccade 5s.csv")
# data = pd.read_csv("/Users/peterliang/Desktop/Neruobit_Python/Result/20200527 B100533006/Report/Horizontal pursuit 5s/Horizontal pursuit 5s.csv")

Fs = 25

Time = timeline(data.Lefteye_X)



input_1_H, index_H, y_3, y_2, y_1,out, input_2, input_3, mean_H, std_H, median_H, iqr_H = SPV_extract(data.Lefteye_X, Fs)

locs, pks, y_f, y_1 = Nystamus_extract(data.Lefteye_X, Fs)

# p_locs, n_locs, pg_locs, ng_locs, p_gain, n_gain, m_gain = pursuit_gain(y_1, input_1_H, index_H, 5)

r, l, a, b, p,n, r_gain, l_gain, p_peak_v, n_peak_v = saccade_gain(y_1, input_1_H, locs, 5)

# print(r_gain, l_gain, p_peak_v, n_peak_v)

plt.plot(Time, y_1,'r', label='raw')
#plt.plot(Time[0:-1], input_1_H,'g', label='input_1')
#plt.plot(Time[0:-1], input_2*100,'r', label='input_2')
#plt.plot(Time[0:-1], input_3,'b', label='input_3')
#plt.plot(index_H/25, y_1[index_H], 'b.',label='y_3')
# plt.plot(p_locs/25, y_1[p_locs], 'b.',label='p')
# plt.plot(n_locs/25, y_1[n_locs], 'y.',label='n')
# plt.plot(pg_locs/25, y_1[pg_locs], 'g.',label='pg')
# plt.plot(ng_locs/25, y_1[ng_locs], 'm.',label='ng')
# plt.plot(r/25, y_1[r], 'b.',label='y_3')
# plt.plot(l/25, y_1[l], 'y.',label='y_3')
# plt.plot(a/25, y_1[a], 'g.',label='y_3')
# plt.plot(b/25, y_1[b], 'm.',label='y_3')
# plt.plot(p_locs/25, y_1[a], 'g.',label='y_3')
# plt.plot(n_locs/25, y_1[b], 'm.',label='y_3')

plt.plot(r/25, y_1[r], 'b.',label='p')
plt.plot(l/25, y_1[l], 'y.',label='n')
plt.plot(a/25, y_1[a], 'g.',label='pf')
plt.plot(b/25, y_1[b], 'm.',label='nf')
plt.plot(p/25, y_1[p], 'c.',label='pg')
plt.plot(n/25, y_1[n], 'k.',label='ng')

plt.legend(loc='upper right')
#plt.plot(p_locs/25, y_1[p_locs], 'g.',label='y_3')
#plt.plot(n_locs/25, y_1[n_locs], 'm.',label='y_3')
# plt.plot(index_H/25, (data.Lefteye_X - np.mean(data.Lefteye_X))[index_H], 'b.',label='y_3')
#plt.plot(index_H/25, y_2[index_H], 'y',label='y_2')
#plt.plot(index_H/25, y_1[index_H], 'm',label='y_1')
#plt.plot(index_H/25, out, 'r', label='out')
plt.show()







