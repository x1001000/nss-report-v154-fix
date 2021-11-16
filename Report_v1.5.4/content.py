# Data
import load_csv as lc
import traceback
import os
# -*- coding: utf-8 -*

REPORT_SECTION = [
    'Patient Table', 'Questionnaire Table', 'Result Table', 'Summary Figure',
    'Dynamic Detail Figure Pur', 'Dynamic Detail Table Pur',
    'Static Detail Figure', 'Static Detail Table',
    'Gaze Table',
    'Dynamic Detail Figure Sin', 'Dynamic Detail Figure Sac',
    'Dynamic Detail Table Sin', 'Dynamic Detail Table Sac',
    'Dynamic Detail Figure OKN', 'Dynamic Detail Table OKN',
]
PATIENT_DATA = [
    'Patient ID', 'Doctor', 'Exam Date', 'Location', 'Division', 'Device', 'Exam'
]

TEST_TYPE = []
for i in range(0, len(lc.patient_test_data)):
    if lc.patient_test_data[i]['file_name'] not in TEST_TYPE:
        TEST_TYPE.append(lc.patient_test_data[i]['file_name'])
TEST_TYPE.sort()

SUMMARY_TYPE = ['Gaze', 'Fixation suppression',
                'Horizontal pursuit 10s', 'Horizontal saccade 10s',
                'Horizontal pursuit 5s', 'Horizontal saccade 5s',
                'Horizontal pursuit 3s', 'Horizontal saccade 3s',
                'Horizontal OKN+ 5s', 'Horizontal OKN- 5s',
                'vHIT']

SUMMARY_THRESHOLD = 50


QUESTIONNAIRE_TYPE = [
    'NIHSS', 'ABCD2', 'DHI-S'
]

RISK_RESULT_TYPE = [
    'Normal', 'Benign disorder', 'Dangerous disorder'
]
RISK_RESULT_THRESHOLD = [ 30, 60 ]
DETAIL_GRAPH_XY_LABELS = [
    'Time (s)', 'Eye Position (°)'
]
DETAIL_GRAPH_CORNER_LABELS = [
    'Right', 'Left', 'Up', 'Down'
]

DYNAMIC_DETAIL_TYPE = [
    'Horizontal pursuit 10s_OD', 'Horizontal pursuit 10s_OS',
    'Horizontal pursuit 5s_OD', 'Horizontal pursuit 5s_OS',
    'Horizontal pursuit 3s_OD', 'Horizontal pursuit 3s_OS',
    'Horizontal pursuit sinusoidal 5s (0.2Hz)_OD', 'Horizontal pursuit sinusoidal 5s (0.2Hz)_OS',
    'Horizontal pursuit sinusoidal 2.5s (0.4Hz)_OD', 'Horizontal pursuit sinusoidal 2.5s (0.4Hz)_OS',
    'Horizontal pursuit sinusoidal 1.25s (0.8Hz)_OD', 'Horizontal pursuit sinusoidal 1.25s (0.8Hz)_OS',
    'Horizontal saccade 10s_OD', 'Horizontal saccade 10s_OS',
    'Horizontal saccade 5s_OD', 'Horizontal saccade 5s_OS',
    'Horizontal saccade 3s_OD', 'Horizontal saccade 3s_OS',
    'Horizontal OKN+ 5s_OD', 'Horizontal OKN+ 5s_OS',
    'Horizontal OKN- 5s_OD', 'Horizontal OKN- 5s_OS'
]

STATIC_DETAIL_TYPE = [
    'Gaze 10s_OD', 'Gaze 10s_OS',
    'Fixation suppression_OD', 'Fixation suppression_OS',
]
DYNAMIC_DETAIL_TABLE_TYPE = {
    'Optokinetic Nystagmus (OKN)' : 'okn',
    'Constant-velocity Pursuit' : 'pursuit',
    'Sinusoidal Pursuit' : 'sin-pursuit',
    'Saccade' : 'saccade'
}
OKN_DATA = {
    'Horizontal SPV' : [ 'Mean', 'Median' ],
    'Vertical SPV': ['Mean', 'Median'],
    'Nystagmus' : [ 'Number', 'Firing Rate (n/10s)']
}

# OKN_TYPE = [
#     'OKN Horizontal (+)', 'OKN Horizontal (-)', 'OKN Vertical (+)' , 'OKN Vertical (-)', 'Asymmetry'
# ]

OKN_TYPE = [
    'OKN Horizontal (+)_OD', 'OKN Horizontal (-)_OD', 'Asymmetry_OD',
    'OKN Horizontal (+)_OS', 'OKN Horizontal (-)_OS', 'Asymmetry_OS',
]

PURSUIT_DATA = {
    'Horizontal SPV' : [ 'Mean', 'Median' ],
    'Vertical SPV': ['Mean', 'Median'],
    'Nystagmus' : [ 'Number', 'Firing Rate (n/10s)'],
    'Pursuit Gain' : [ 'Rightward Gain (med.)', 'Leftward Gain (med.)', 'Mean Gain (med.)' ]
}

# PURSUIT_TYPE = [
#     'Horizontal pursuit 10s', 'Horizontal pursuit 5s', 'Horizontal pursuit 3s',
#     'Vertical pursuit 10s', 'Vertical pursuit 5s', 'Vertical pursuit 3s'
# ]

PURSUIT_TYPE = [
    'Horizontal pursuit 10s_OD', 'Horizontal pursuit 10s_OS',
    'Horizontal pursuit 5s_OD', 'Horizontal pursuit 5s_OS',
    'Horizontal pursuit 3s_OD', 'Horizontal pursuit 3s_OS',
]

# SIN_PURSUIT_TYPE = [
#     'Horizontal pursuit 10s (Sin.)', 'Horizontal pursuit 5s (Sin.)', 'Horizontal pursuit 3s (Sin.)',
#     'Vertical pursuit 10s (Sin.)', 'Vertical pursuit 5s (Sin.)', 'Vertical pursuit 3s (Sin.)'
# ]

SIN_PURSUIT_TYPE = [
    'Horizontal pursuit sinusoidal 5s (0.2Hz)_OD', 'Horizontal pursuit sinusoidal 5s (0.2Hz)_OS',
    'Horizontal pursuit sinusoidal 2.5s (0.4Hz)_OD', 'Horizontal pursuit sinusoidal 2.5s (0.4Hz)_OS',
    'Horizontal pursuit sinusoidal 1.25s (0.8Hz)_OD', 'Horizontal pursuit sinusoidal 1.25s (0.8Hz)_OS',
]

SACCADE_DATA = {
    'Horizontal SPV' : [ 'Mean', 'Median' ],
    'Vertical SPV' : [ 'Mean', 'Median' ],
    'Nystagmus' : [ 'Number', 'Firing Rate (n/10s)'],
    'Saccadic Accuracy' : [ 'Rightward Gain (med.)', 'Leftward Gain (med.)', 'Mean Gain (med.)'],
    'Peak Velocity' : [ 'Rightward (med.)', 'Leftward (med.)', 'Mean PV (med.)' ]
}

# SACCADE_TYPE = [
#     'Horizontal saccade 10s', 'Horizontal saccade 5s', 'Horizontal saccade 3s',
#     'Vertical saccade 10s', 'Vertical saccade 5s', 'Vertical saccade 3s'
# ]

SACCADE_TYPE = [
    'Horizontal saccade 10s_OD', 'Horizontal saccade 10s_OS',
    'Horizontal saccade 5s_OD', 'Horizontal saccade 5s_OS',
    'Horizontal saccade 3s_OD', 'Horizontal saccade 3s_OS',
]

STATIC_DETAIL_TABLE_TYPE = {
    'Gaze 10s' : 'gaze', 'Fixation suppression' : 'fix'
}
GAZE_DATA = {
    'Horizontal SPV' : [ 'Mean', 'Median' ],
    'Vertical SPV': ['Mean', 'Median'],
    'Nystagmus' : [ 'Number', 'Firing Rate (n/10s)']
}
GAZE_TYPE = [
    'All_OD',
    'Center_OD', 'Up_OD', 'Right_OD', 'Down_OD', 'Left_OD',
    'All_OS',
    'Center_OS', 'Up_OS', 'Right_OS', 'Down_OS', 'Left_OS',
]

FIX_DATA = {
    'Horizontal SPV' : [ 'Mean', 'Median' ],
    'Vertical SPV': ['Mean', 'Median'],
    'Nystagmus' : [ 'Number', 'Firing Rate (n/10s)']
}

FIX_TYPE = [
    'Fixation suppression_OD', 'Fixation suppression_OS', #suppression / non-suppression
]

DYNAMIC_DETAIL_LABELS = [
    'Horizontal', 'Vertical', "Target_Horizontal", "Target_Vertical"
]


# Ranges and Units
RANGE_STD = ' ± Std'
RANGE_IQR = ' ± IQR'
UNIT_S = ' (s)'
UNIT_DEGREE = '  (°)'

# Titles
TITLE = 'Vestibular-Oculo-Reflex Function and Brain Risk Detection Report'
SUMMARY_TITLE = '<b>SUMMARY</b>'
SUMMARY_FIG_TITLE = '<b>FIGURE</b>'
SUMMARY_EX_TITLE = '<b>EXAMPLE</b>'
DYNAMIC_DETAILS_TITLE = '<b>DYNAMIC EXAMINATION REPORT DETAILS</b>'
STATIC_DETAILS_TITLE = '<b>STATIC EXAMINATION REPORT DETAILS</b>'
GAZE_TITLE = '<b>9 GAZE</b>'

# Text
QUESTIONNAIRE = 'Multimodal Questionnaire: '
RISK_RESULT = 'Result of Risk: '
DISCLAIMER = 'Report printed from Neurobit Technologies Co., Ltd. For assisted diagnosis service and general assessment only.'
DYNAMIC_LABELS = "Labels"

# Figures
SUMMARY_FIG_L = 'summary_l.png' # stroke-problems
SUMMARY_FIG_R = 'summary_r.png' # stroke-problems
SUMMARY_FIG_N = 'summary_n.png' # stroke-problems
SUMMARY_FIG_B = 'summary_b.png' # stroke-problems
SUMMARY_FIG_D = 'summary_d.png' # stroke-problems
DYNAMIC_DETAIL_FIG_PUR = 'dynamic_detail_pur.svg'
DYNAMIC_DETAIL_FIG_SIN = 'dynamic_detail_sin.svg'
DYNAMIC_DETAIL_FIG_SAC = 'dynamic_detail_sac.svg'
DYNAMIC_DETAIL_FIG_OKN = 'dynamic_detail_okn.svg'
STATIC_DETAIL_FIG = 'static_detail.svg'
LOGO_IMAGE = 'logo.png'

