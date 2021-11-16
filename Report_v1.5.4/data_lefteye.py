### TEMPORARY - for testing uses

import os
project_path = os.getcwd()
error_path = os.path.join(project_path, "Error_logs")

### Processing Index
import content as mcontent
import load_csv as lc
# import eye_track
import SPV_main as SPV
import math

os.chdir(project_path)

### DATA

patient_ID = lc.patient_ID
# result_path = os.path.join(lc.data_path, lc.patient_ID)
result_path = os.path.join(lc.patient_test_dir_path, "Report")
report_name = patient_ID + ".pdf"

try:
    eye_track_rec = eye_track.record
except:
    print("Eye-Tracking function is not opened.")

def test_data():
    
    data = {}
    data[ mcontent.REPORT_SECTION[0] ] = patient_data()
    data[ mcontent.REPORT_SECTION[1] ] = questionnaire_data()
    data[ mcontent.REPORT_SECTION[2] ] = result_data()
    data[ mcontent.REPORT_SECTION[3] ] = summary_data()
    data[ mcontent.REPORT_SECTION[4] ] = dynamic_detail_data_pur()
    data[ mcontent.REPORT_SECTION[5] ] = dynamic_table_data()
    data[ mcontent.REPORT_SECTION[6] ] = static_detail_data()
    data[ mcontent.REPORT_SECTION[7] ] = static_table_data()
    data[ mcontent.REPORT_SECTION[8] ] = gaze_data()
    data[ mcontent.REPORT_SECTION[9] ] = dynamic_detail_data_sin()
    data[ mcontent.REPORT_SECTION[10] ] = dynamic_detail_data_sac()
    data["POS"] = "Left"

    return data

def summary_data():
    data = {}
    max_r = 0
    min_r = 100
    range_r = 0
    max_iqr = 0
    ratio_iqr = 1

    # calculate ratio
    for i in range(0, len(lc.patient_test_data)):
        try:
            if abs(SPV.record[i]['l_mean_H']) > max_r:
                max_r = abs(SPV.record[i]['l_mean_H'])
            if abs(SPV.record[i]['l_mean_H']) <= min_r:
                min_r = abs(SPV.record[i]['l_mean_H'])
            if SPV.record[i]['l_iqr_H'] >= max_iqr:
                max_iqr = SPV.record[i]['l_iqr_H']
        except:
            print("summary_data occured ERROR_1.")

        try:
            # print(SPV.record[i]['l_mean_H'])
            # print(max_r, min_r)
            range_r = max_r - min_r

            # modify value of IQR ratio in graph
            max_iqr = max_iqr * 10 # MARKSIZE = 10
            if max_iqr >= 100:
                ratio_iqr = 100 / max_iqr
        except:
            print("summary_data occured ERROR_2.")

    for i in range(0, len(lc.patient_test_data)):

        '''
        data = {
            'Calibration' : {
                'Value' : 60,
                'Vertical' : [ -1.2, -0.8, 0, 0.3, 0.9 ],
                'Horizontal' : [ -0.8, -0.5, 0, 0.6, 1.1 ]
            } , 'Gaze Evoke' : {
                'Value' : 30,
                'Vertical' : [ -0.6, -0.5, 0, 0.3, 0.9 ],
                'Horizontal' : [ -1.3, -0.75, 0, 0.6, 1.6 ]
            }
        }
        '''
        try:
            data.update({
                lc.patient_test_data[i]['file_name']:{
                    'Value': math.sqrt((abs(SPV.record[i]['l_mean_H']) - min_r) / range_r * 100) * 10,
                    'Vertical': [(-SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                 (-0.5 * SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                  SPV.record[i]['l_mean_H'],
                                 (0.5 * SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                 (SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                 ],
                    'Horizontal': [(-SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                   (-0.5 * SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                    SPV.record[i]['l_mean_H'],
                                   (0.5 * SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                   (SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                   ]
                }
                })

            if "Gaze" in lc.patient_test_data[i]['file_name']:
                # n = " ".join(lc.patient_test_data[i]['file_name'].split(" ")[0:2])
                n = lc.patient_test_data[i]['file_name'].split(" ")[0]
                #print(n)
                data.update({
                    n :{
                        'Value': math.sqrt((abs(SPV.record[i]['l_mean_H']) - min_r) / range_r * 100) * 10,
                        'Vertical': [(-SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                     (-0.5 * SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                      SPV.record[i]['l_median_H'],
                                     (0.5 * SPV.record[i]['l_iqr_V']),
                                     (SPV.record[i]['l_iqr_V']),
                                     ],
                        'Horizontal': [(-SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                       (-0.5 * SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                        SPV.record[i]['l_median_H'],
                                       (0.5 * SPV.record[i]['l_iqr_H']) * ratio_iqr,
                                       (SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                       ]
                    }
                })

            elif "Fixation" in lc.patient_test_data[i]['file_name']:
                # fix = lc.patient_test_data[i]['file_name'].split(" ")[0]
                fix = " ".join(lc.patient_test_data[i]['file_name'].split(" ")[0:2])
                #print(fix)
                data.update({
                    fix :{
                        'Value': math.sqrt((abs(SPV.record[i]['l_mean_H']) - min_r) / range_r * 100) * 10,
                        'Vertical': [(-SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                     (-0.5 * SPV.record[i]['l_iqr_V'] * ratio_iqr),
                                      SPV.record[i]['l_median_H'],
                                     (0.5 * SPV.record[i]['l_iqr_V']),
                                     (SPV.record[i]['l_iqr_V']),
                                     ],
                        'Horizontal': [(-SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                       (-0.5 * SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                        SPV.record[i]['l_median_H'],
                                       (0.5 * SPV.record[i]['l_iqr_H']) * ratio_iqr,
                                       (SPV.record[i]['l_iqr_H'] * ratio_iqr),
                                       ]
                    }
                })
        except:
            print("summary_data occured ERROR_3.")

    return data

def patient_data():
    data = {
        'Patient ID' : lc.patient_test_data[0]['Name'][0],
        'Doctor' : lc.patient_test_data[0]['Doctor'][0],
        'Location' : 'N/A',
        'Exam Date' : lc.patient_test_data[0]['Date'][0],
        'Division' : 'Neurology', 
        'Device' : lc.patient_test_data[0]['Device'][0],
        'Exam' : mcontent.TEST_TYPE
    }
    return data

def questionnaire_data():
    try:
        data = {
            'NIHSS' : int(lc.patient_test_data[0]['NIHSS'][0]),
            #'ABCD3i' : int(lc.patient_test_data[0]['ABCD3i'][0]),
            'ABCD2': int(lc.patient_test_data[0]['ABCD2'][0]),
            'DHI-S' : int(lc.patient_test_data[0]['DHI_S'][0])
        }
    except:
        data = {
            'NIHSS': int(lc.patient_test_data[0]['NIHSS'][0]),
            #'ABCD3i' : int(lc.patient_test_data[0]['ABCD3i'][0]),
            'ABCD2': int(lc.patient_test_data[0]['ABCD2'][0]),
            'DHI-S': int(lc.patient_test_data[0]['DHI_S'][0])
        }

    return data

def result_data():
    N = B = D = 0

    for i in range(0, len(lc.patient_test_data)):
        # V
        if "Vertical" in lc.patient_test_data[i]['file_name']:
            try:
                if abs(SPV.record[i]['l_mean_V']) < 1:
                    N += 1
                elif abs(SPV.record[i]['l_mean_V']) >= 1 and abs(SPV.record[i]['l_mean_V']) < 3:
                    B += 1
                elif abs(SPV.record[i]['l_mean_V']) >= 3:
                    D += 1
            except:
                print("Vertical result_data occured ERROR.")
        # H
        else:
            try:
                if abs(SPV.record[i]['l_mean_H']) < 1:
                    N += 1
                elif abs(SPV.record[i]['l_mean_H']) >= 1 and abs(SPV.record[i]['l_mean_H']) < 3:
                    B += 1
                elif abs(SPV.record[i]['l_mean_H']) >= 3:
                    D += 1
            except:
                print("Horizontal result_data occured ERROR.")

        S = N + B + D

    # print(S)

    try:
        Normal_Score = round(N/S, 4) * 100
        Benign_Score = round(B/S, 4) * 100
        Dangerous_Score = round(D/S, 4) * 100

        Diagnosis = ""

        if max(Normal_Score, Benign_Score, Dangerous_Score) == Dangerous_Score:
            Diagnosis = 'Dangerous disorder'
        elif max(Normal_Score, Benign_Score, Dangerous_Score) == Benign_Score:
            Diagnosis = 'Benign disorder'
        elif max(Normal_Score, Benign_Score, Dangerous_Score) == Normal_Score:
            Diagnosis = 'Normal'

        data = {
            'Normal' : Normal_Score, 'Benign disorder' : Benign_Score, 'Dangerous disorder' : Dangerous_Score,
            'Diagnosis' : Diagnosis,
        }
    except:
        data = {
            'Normal': 0, 'Benign disorder': 0, 'Dangerous disorder': 0,
            'Diagnosis': 'N/A',
        }

        print("result_data() occured ERROR.")

    return data

def dynamic_detail_data_pur():
    data = {}
    for i in range(0, len(lc.patient_test_data)):
        if "Horizontal pursuit" in lc.patient_test_data[i]['file_name']:
            data.update({
                lc.patient_test_data[i]['file_name'] : {
                    'Horizontal': (SPV.record[i]['Time_E'], SPV.record[i]['L_X']),
                    'Vertical' : ( SPV.record[i]['Time_E'], SPV.record[i]['L_Y'] ),
                    'Target_Horizontal' : ( SPV.record[i]['Time'], SPV.record[i]['spot_x_angle'] ),
                    'Target_Vertical' : ( SPV.record[i]['Time'], SPV.record[i]['spot_y_angle'] )
                }
            })
    return data

def dynamic_detail_data_sin():
    data = {}
    for i in range(0, len(lc.patient_test_data)):
        if "Horizontal pursuit sin" in lc.patient_test_data[i]['file_name']:
            data.update({
                lc.patient_test_data[i]['file_name'] : {
                    'Horizontal': (SPV.record[i]['Time_E'], SPV.record[i]['L_X']),
                    'Vertical' : ( SPV.record[i]['Time_E'], SPV.record[i]['L_Y'] ),
                    'Target_Horizontal' : ( SPV.record[i]['Time'], SPV.record[i]['spot_x_angle'] ),
                    'Target_Vertical' : ( SPV.record[i]['Time'], SPV.record[i]['spot_y_angle'] )
                }
            })
    return data

def dynamic_detail_data_sac():
    data = {}
    for i in range(0, len(lc.patient_test_data)):
        if "Horizontal saccade" in lc.patient_test_data[i]['file_name']:
            data.update({
                lc.patient_test_data[i]['file_name'] : {
                    'Horizontal': (SPV.record[i]['Time_E'], SPV.record[i]['L_X']),
                    'Vertical' : ( SPV.record[i]['Time_E'], SPV.record[i]['L_Y'] ),
                    'Target_Horizontal' : ( SPV.record[i]['Time'], SPV.record[i]['spot_x_angle'] ),
                    'Target_Vertical' : ( SPV.record[i]['Time'], SPV.record[i]['spot_y_angle'] )
                }
            })
    return data

def static_detail_data():
    data = {}
    for i in range(0, len(lc.patient_test_data)):
        if "Gaze" in lc.patient_test_data[i]['file_name']:
            data.update({
                lc.patient_test_data[i]['file_name'].split(" ")[0]: {
                    'Horizontal': (SPV.record[i]['Time_E'], SPV.record[i]['L_X']),
                    'Vertical': (SPV.record[i]['Time_E'], SPV.record[i]['L_Y']),
                    'Target_Horizontal': (SPV.record[i]['Time'], SPV.record[i]['spot_x_angle']),
                    'Target_Vertical': (SPV.record[i]['Time'], SPV.record[i]['spot_y_angle'])
                }
            })
        elif "Fixation" in lc.patient_test_data[i]['file_name']:
            fix = " ".join(lc.patient_test_data[i]['file_name'].split(" ")[0:2])
            data.update({
                fix: {
                    'Horizontal': (SPV.record[i]['Time_E'], SPV.record[i]['L_X']),
                    'Vertical': (SPV.record[i]['Time_E'], SPV.record[i]['L_Y']),
                }
            })

    # 原本資料寫入形式
    '''
    data = {
        'Gaze saccade' : {
            'Vertical' : ( [0, 30], [0, 0] ),
            'Horizontal' : ( [0, 30], [-1, -1] )
        }, 'Fixation Suppression' : {
            'Vertical' : ( [0, 30], [0, 0] ),
            'Horizontal' : ( [0, 30], [-1, -1] )
        }
    }
    '''
    return data

def dynamic_table_data():
    # test types
    OKN_type = {}
    Tri_pursuit_type = {}
    Sin_pursuit_type = {}
    Saccade_type = {}
    data = {'Optokinetic Nystagmus (OKN)': OKN_type,
            'Constant-velocity Pursuit': Tri_pursuit_type,
            'Sinusoidal Pursuit': Sin_pursuit_type,
            'Saccade': Saccade_type
            }
    # OKN record
    OKN_ph_mean = 0
    OKN_ph_med = 0
    OKN_nh_mean = 0
    OKN_nh_med = 0

    OKN_pv_mean = 0
    OKN_pv_med = 0
    OKN_nv_mean = 0
    OKN_nv_med = 0



    for i in range(0, len(lc.patient_test_data)):
        ## OKN part
        if "Horizontal OKN+" in lc.patient_test_data[i]['file_name']:
            OKN_ph_mean = SPV.record[i]['l_mean_H']
            OKN_ph_med = SPV.record[i]['l_median_H']
            OKN_pv_mean = SPV.record[i]['l_mean_V']
            OKN_pv_med = SPV.record[i]['l_median_V']
            OKN_type.update({
                    'OKN Horizontal (+)': {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_H'], SPV.record[i]['l_std_H']),
                                'Median': (SPV.record[i]['l_median_H'], SPV.record[i]['l_iqr_H'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR'] }
                    }
            })
            if OKN_nh_mean != 0 and OKN_nh_med != 0 and OKN_pv_mean != 0 and OKN_pv_med != 0:
                try:
                    OKN_type.update({
                        'Asymmetry': {
                            'Horizontal SPV': {'Mean': (round((OKN_ph_mean-OKN_nh_mean) / (OKN_ph_mean+OKN_nh_mean), 2)),
                                    'Median': (round((OKN_ph_med-OKN_nh_med) / (OKN_ph_med+OKN_nh_med), 2))},
                            'Vertical SPV': {'Mean': (round((OKN_pv_mean-OKN_nv_mean) / (OKN_pv_mean+OKN_nv_mean), 2)),
                                    'Median': (round((OKN_pv_med-OKN_nv_med) / (OKN_pv_med+OKN_nv_med), 2))},
                            'Nystagmus': {'Number': "-", 'Firing Rate': "-"}
                        }
                    })
                except:
                    OKN_type.update({
                        'Asymmetry': {
                            'Horizontal SPV': {'Mean': "N/A",
                                    'Median': "N/A"},
                            'Vertical SPV': {'Mean': "N/A",
                                    'Median': "N/A"},
                            'Nystagmus': {'Number': "-", 'Firing Rate': "-"}
                        }
                    })



        elif "Horizontal OKN-" in lc.patient_test_data[i]['file_name']:
            OKN_nh_mean = SPV.record[i]['l_mean_H']
            OKN_nh_med = SPV.record[i]['l_median_H']
            OKN_nv_mean = SPV.record[i]['l_mean_V']
            OKN_nv_med = SPV.record[i]['l_median_V']

            OKN_type.update({
                    'OKN Horizontal (-)': {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_H'], SPV.record[i]['l_std_H']),
                                'Median': (SPV.record[i]['l_median_H'], SPV.record[i]['l_iqr_H']) },
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']}
                    }
            })
            if OKN_nh_mean != 0 and OKN_nh_med != 0:
                try:
                    OKN_type.update({
                        'Asymmetry': {
                            'Horizontal SPV': {'Mean': (round((OKN_ph_mean - OKN_nh_mean) / (OKN_ph_mean + OKN_nh_mean), 2)),
                                    'Median': (round((OKN_ph_med - OKN_nh_med) / (OKN_ph_med + OKN_nh_med), 2))},
                            'Vertical SPV': {
                                'Mean': (round((OKN_pv_mean - OKN_nv_mean) / (OKN_pv_mean + OKN_nv_mean), 2)),
                                'Median': (round((OKN_pv_med - OKN_nv_med) / (OKN_pv_med + OKN_nv_med), 2))},
                            'Nystagmus': {'Number': "-", 'Firing Rate': "-"}
                        }
                    })
                except:
                    OKN_type.update({
                        'Asymmetry': {
                            'Horizontal SPV': {'Mean': "N/A",
                                    'Median': "N/A"},
                            'Vertical SPV': {'Mean': "N/A",
                                             'Median': "N/A"},
                            'Nystagmus': {'Number': "-", 'Firing Rate': "-"}
                        }
                    })
        elif "Vertical OKN+ " in lc.patient_test_data[i]['file_name']:
            OKN_type.update({
                    'OKN Horizontal (+)': {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR'] }
                    }
            })

        elif "Vertical OKN-" in lc.patient_test_data[i]['file_name']:
            OKN_type.update({
                    'OKN Horizontal (-)': {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V']) },
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']}
                    }
            })

        ## Triangle pursuit part
        elif "Horizontal pursuit" in lc.patient_test_data[i]['file_name'] and not lc.patient_test_data[i]['Sine'][0]:
            Tri_pursuit_type.update({
                    lc.patient_test_data[i]['file_name']: {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_H'], SPV.record[i]['l_std_H']),
                                'Median': (SPV.record[i]['l_median_H'], SPV.record[i]['l_iqr_H'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']},
                        'Pursuit Gain': {'Rightward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pp_gain'], SPV.record[i]['l_pp_med']) ,
                                         'Leftward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pn_gain'], SPV.record[i]['l_pn_med']),
                                         'Mean Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pm_gain'], SPV.record[i]['l_pm_med'])}
                    }
            })
            # print(Tri_pursuit_type)
        elif "Vertical pursuit" in lc.patient_test_data[i]['file_name'] and not lc.patient_test_data[i]['Sine'][0]:
            Tri_pursuit_type.update({
                    lc.patient_test_data[i]['file_name']: {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']},
                        'Pursuit Gain': {'Rightward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pp_gain'], SPV.record[i]['l_pp_med']) ,
                                         'Leftward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pn_gain'], SPV.record[i]['l_pn_med']),
                                         'Mean Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pm_gain'], SPV.record[i]['l_pm_med'])}
                    }
            })

        ## Sinusoidal Pursuit part
        elif "Horizontal pursuit" in lc.patient_test_data[i]['file_name'] and lc.patient_test_data[i]['Sine'][0]:
            Sin_pursuit_type.update({
                    lc.patient_test_data[i]['file_name']: {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_H'], SPV.record[i]['l_std_H']),
                                'Median': (SPV.record[i]['l_median_H'], SPV.record[i]['l_iqr_H'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']},
                        'Pursuit Gain': {'Rightward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pp_gain'], SPV.record[i]['l_pp_med']) ,
                                         'Leftward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pn_gain'], SPV.record[i]['l_pn_med']),
                                         'Mean Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pm_gain'], SPV.record[i]['l_pm_med'])
                                         }
                    }
            })
        elif "Vertical pursuit" in lc.patient_test_data[i]['file_name'] and lc.patient_test_data[i]['Sine'][0]:
            Sin_pursuit_type.update({
                    lc.patient_test_data[i]['file_name']: {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']},
                        'Pursuit Gain': {'Rightward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pp_gain'], SPV.record[i]['l_pp_med']) ,
                                         'Leftward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pn_gain'], SPV.record[i]['l_pn_med']),
                                         'Mean Gain (med.)': "%s (%s)" % (SPV.record[i]['l_pm_gain'], SPV.record[i]['l_pm_med'])}
                    }
            })
                

        ## Saccade part
        elif "Horizontal saccade" in lc.patient_test_data[i]['file_name']:
            Saccade_type.update({
                    lc.patient_test_data[i]['file_name']: {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_H'], SPV.record[i]['l_std_H']),
                                'Median': (SPV.record[i]['l_median_H'], SPV.record[i]['l_iqr_H'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']},
                        'Saccadic Accuracy': {'Rightward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_sp_gain'], SPV.record[i]['l_sp_med']),
                                         'Leftward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_sn_gain'], SPV.record[i]['l_sn_med']),
                                         'Mean Gain (med.)': "%s (%s)" % (SPV.record[i]['l_sm_gain'], SPV.record[i]['l_sm_med'])},
                        'Peak Velocity': {'Rightward (med.)': "%s (%s)" % (SPV.record[i]['l_spv_p'], SPV.record[i]['l_spv_p_med']),
                                          'Leftward (med.)': "%s (%s)" % (SPV.record[i]['l_spv_n'], SPV.record[i]['l_spv_n_med']),
                                          'Mean PV (med.)': "%s (%s)" % (SPV.record[i]['l_spv_m'], SPV.record[i]['l_spv_m_med'])}
                    }
            })
            # print(Saccade_type)

        elif "Vertical saccade" in lc.patient_test_data[i]['file_name']:
            Saccade_type.update({
                    lc.patient_test_data[i]['file_name']: {
                        'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                         'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                        'Nystagmus': {'Number': SPV.record[i]['l_sc_n'],  'Firing Rate': SPV.record[i]['l_FR']},
                        'Saccadic Accuracy': {'Rightward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_sp_gain'], SPV.record[i]['l_sp_med']),
                                         'Leftward Gain (med.)': "%s (%s)" % (SPV.record[i]['l_sn_gain'], SPV.record[i]['l_sn_med']),
                                         'Mean Gain (med.)': "%s (%s)" % (SPV.record[i]['l_sm_gain'], SPV.record[i]['l_sm_med'])},
                        'Peak Velocity': {'Rightward (med.)': "%s (%s)" % (SPV.record[i]['l_spv_p'], SPV.record[i]['l_spv_p_med']),
                                          'Leftward (med.)': "%s (%s)" % (SPV.record[i]['l_spv_n'], SPV.record[i]['l_spv_n_med']),
                                          'Mean PV (med.)': "%s (%s)" % (SPV.record[i]['l_spv_m'], SPV.record[i]['l_spv_m_med'])}
                    }
            })

    # report-v1
    '''
    data = {
        'Optokinetic Nystagmus (OKN)' : {
            'OKN Horizontal (+)' : {
                'Horizontal SPV' : { 'Mean' : (5.53, 2.1), 'Median' : (-0.8, 0.7) },
                'Nystagmus' : { 'Number' : 49, 'Firing Rate': 50 }
            }, 
            'OKN Horizontal (-)' : {
                'Horizontal SPV' : { 'Mean' : (5.53, 2.1), 'Median' : (-0.8, 0.7) },
                'Nystagmus' : { 'Number' : 49, 'Firing Rate': 50 }
            },
        },
        'Triangle' : {
            'Horizontal Pursuit 10s' : {
                'Pursuit Gain' : { 'Mean' : (0.35, 0.4), 'Median' : (0.38, 0.7)  }
            },
            'Horizontal Pursuit 3s' : {
                'Pursuit Gain' : { 'Mean' : (0.4, 0.3), 'Median' : (0.34, 0.12)  }
            }
        },
        'Saccade' : {
            'Horizontal Saccade 5s' : {
                'Saccade' : { 'Accuracy' : 0.4, 'Latency' : (0.2, 0.15), 'Amplitude' : (0.8, 0.05) }
            }, 'Horizontal Saccade 3s' : {
                'Saccade' : { 'Accuracy' : 0.75, 'Latency' : (0.1, 0.05), 'Amplitude' : (0.3, 0.05) }
            }
        }
    }
    '''

    return data

def static_table_data():
    # test types
    Gaze_type = {}
    Fixation_type = {}
    data = {'Gaze': Gaze_type,
            'Fixation suppression': Fixation_type
            }

    for i in range(0, len(lc.patient_test_data)):
        ## Gaze evoked part
        if "Gaze" in lc.patient_test_data[i]['file_name']:
            Gaze_type.update({
                'Center (Hor.)': {
                    'SPV': {'Mean': (SPV.record[i]['l_ge_c_mean'], SPV.record[i]['l_ge_c_std']),
                            'Median': (SPV.record[i]['l_ge_c_median'], SPV.record[i]['l_ge_c_iqr'])},
                    'Nystagmus': {'Number': SPV.record[i]['l_ge_c_st'], 'Firing Rate': SPV.record[i]['l_ge_c_fr']}
                },
                'Up (Ver.)': {
                    'SPV': {'Mean': (SPV.record[i]['l_ge_u_mean'], SPV.record[i]['l_ge_u_std']),
                            'Median': (SPV.record[i]['l_ge_u_median'], SPV.record[i]['l_ge_u_iqr'])},
                    'Nystagmus': {'Number': SPV.record[i]['l_ge_u_st'], 'Firing Rate': SPV.record[i]['l_ge_u_fr']}
                },
                'Right (Hor.)': {
                    'SPV': {'Mean': (SPV.record[i]['l_ge_r_mean'], SPV.record[i]['l_ge_r_std']),
                            'Median': (SPV.record[i]['l_ge_r_median'], SPV.record[i]['l_ge_r_iqr'])},
                    'Nystagmus': {'Number': SPV.record[i]['l_ge_r_st'], 'Firing Rate': SPV.record[i]['l_ge_r_fr']}
                },
                'Down (Ver.)': {
                    'SPV': {'Mean': (SPV.record[i]['l_ge_d_mean'], SPV.record[i]['l_ge_d_std']),
                            'Median': (SPV.record[i]['l_ge_d_median'], SPV.record[i]['l_ge_d_iqr'])},
                    'Nystagmus': {'Number': SPV.record[i]['l_ge_d_st'], 'Firing Rate': SPV.record[i]['l_ge_d_fr']}
                },
                'Left (Hor.)': {
                    'SPV': {'Mean': (SPV.record[i]['l_ge_l_mean'], SPV.record[i]['l_ge_l_std']),
                            'Median': (SPV.record[i]['l_ge_l_median'], SPV.record[i]['l_ge_l_iqr'])},
                    'Nystagmus': {'Number': SPV.record[i]['l_ge_l_st'], 'Firing Rate': SPV.record[i]['l_ge_l_fr']}
                }
            })
            #print(Gaze_type)

        elif "Fixation" in lc.patient_test_data[i]['file_name']:
            Fixation_type.update({
                'Fixation suppression': {
                    'Horizontal SPV': {'Mean': (SPV.record[i]['l_mean_H'], SPV.record[i]['l_std_H']),
                            'Median': (SPV.record[i]['l_median_H'], SPV.record[i]['l_iqr_H'])},
                    'Vertical SPV': {'Mean': (SPV.record[i]['l_mean_V'], SPV.record[i]['l_std_V']),
                                     'Median': (SPV.record[i]['l_median_V'], SPV.record[i]['l_iqr_V'])},
                    'Nystagmus': {'Number': SPV.record[i]['l_sc_n'], 'Firing Rate': SPV.record[i]['l_FR']}
                }
            })

    # data = {
    #     'Fixation' : {
    #         'Positive' : {
    #             'SPV' : { 'Mean' : (5.53, 2.1), 'Median' : (-0.8, 0.7) },
    #             'Nystagmus' : { 'Number' : 49, 'Duration' : (0.42, 0.09), 'Peak Velocity' : (10, 2), 'Amplitude' : (5, 2.1) }
    #         },
    #         'Negative' : {
    #             'SPV' : { 'Mean' : (5.53, 2.1), 'Median' : (-0.8, 0.7) },
    #             'Nystagmus' : { 'Number' : 49, 'Duration' : (0.42, 0.09), 'Peak Velocity' : (10, 2), 'Amplitude' : (5, 2.1) }
    #         }
    #     },
    #     'Gaze' : {
    #         'Up' : {
    #             'Saccade' : { 'Accuracy' : 0.4, 'Latency' : (0.2, 0.15), 'Amplitude' : (0.8, 0.05) }
    #         }, 'Down' : {
    #             'Saccade' : { 'Accuracy' : 0.75, 'Latency' : (0.1, 0.05), 'Amplitude' : (0.3, 0.05) }
    #         }
    #     }
    # }
    return data

def gaze_data():
    data = []
    gaze_fig = 'gaze_%d.png'
    for i in range(1, 10):
        data.append(gaze_fig % i)
    return data

### FUNCTIONAL

def summary_data_by_median(summary_data):

    # vertical
    for t in mcontent.SUMMARY_TYPE:
        if t in summary_data:
            median_v = summary_data[t]['Vertical'][2]
            median_h = summary_data[t]['Horizontal'][2]

            '''
            for i in range( len(summary_data[t]['Vertical']) ):
                summary_data[t]['Vertical'][i] = summary_data[t]['Vertical'][i] - median_v
            
            for j in range( len(summary_data[t]['Horizontal']) ):
                summary_data[t]['Horizontal'][j] = summary_data[t]['Horizontal'][j] - median_h
            '''

    return summary_data
