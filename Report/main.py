import content as mcontent
import summary_graph as msummary
import patient_table as mpatient
import questionnaire_table as mquestionnaire
import result_table as mresult
import detail_graph as mdetail
import detail_table as mdetailtable
import data_righteye as r_mdata
import data_lefteye as l_mdata
import dynamic_detail_labels as mlabels
from PyPDF2 import PdfFileWriter, PdfFileReader
import random

# parameters
import load_csv as lc
from SPV_main import record
logs = ""

import sys
from PIL import Image as pImage
import os
import shutil
import errno
# import mdbtools
import subprocess

# ReportLab imports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib import colors
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

# logging
import sys
import logging
import traceback
import csv

H_MARGIN = 20
T_MARGIN = 30
B_MARGIN = 18
P_MARGIN = 10

CHART_TEXT_SIZE = 7

def initialize_PDF(file_name):
    '''
    Initialize PDF

    file_name - filename for PDF report; will make directories if necessary
    returns - doc object 
    '''
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print('Directory exists')
    doc = SimpleDocTemplate(file_name, pagesize=A4,
                        rightMargin=H_MARGIN,leftMargin=H_MARGIN,
                        topMargin=T_MARGIN, bottomMargin=B_MARGIN)
    return doc

def content(doc, report_data_r, report_data_l):
    story = []
    
    styles = getSampleStyleSheet()
    bodyTextStyle = styles['BodyText']
    sectionTitleStyle = ParagraphStyle('default', 
                            alignment = TA_LEFT,
                            firstLineIndent = 0,
                            leftIndent = 0,
                            spaceAfter= 10)
    chartTitleStyle = ParagraphStyle('default', 
                            alignment = TA_LEFT,
                            fontSize=12)
    headerStyle = ParagraphStyle('default', 
                            alignment = TA_CENTER,
                            fontSize=CHART_TEXT_SIZE,
                            textColor=colors.white)

    # Title
    title = Paragraph( mcontent.TITLE, styles['Heading2'] )

    # Patient data
    patient_content = mpatient.content( report_data_r[ mcontent.REPORT_SECTION[0] ] )
    for i in range( len(patient_content) ):
        for j in range( len(patient_content[i]) ):
            patient_content[i][j] = Paragraph( patient_content[i][j], bodyTextStyle )
    patient_table = Table( patient_content, colWidths=(A4[0] - H_MARGIN * 2)/3, spaceAfter=P_MARGIN )
    patient_table.setStyle( TableStyle(
        [('ALIGN', (0, 0), (-1, -1),'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('SPAN', (0, 0), (0, 1)),
        ('SPAN', (0, 2), (2, 2)),
        ('SPAN', (0, 3), (2, 3)),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black)]) )

    # Multimodal questionnaire
    questionnaire_content = mquestionnaire.content( report_data_r[ mcontent.REPORT_SECTION[1] ] )
    questionnaire_table = Table( questionnaire_content, colWidths=(A4[0] - H_MARGIN * 2)/4, spaceAfter=P_MARGIN )
    questionnaire_table.setStyle( TableStyle(
        [('ALIGN', (0, 0), (0, 0),'LEFT'),
        ('ALIGN', (1, 0), (-1, -1),'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEABOVE', (1, 0), (-1, 0), 0.5 , colors.black),
        ('LINEBELOW', (1, 0), (-1, 0), 0.5 , colors.black)]) )

    # Results
    result_content = mresult.content(report_data_r[ mcontent.REPORT_SECTION[2] ])
    result_table = Table( result_content, spaceAfter=P_MARGIN, hAlign='CENTER' )
    result_table.setStyle( TableStyle(
        [('ALIGN', (0, 0), (-1, -1),'LEFT'),
        ('SPAN', (0, 0), (0, -1)),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (0, 0), 15),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')]) )


    # Summary figure
    summary_title = Paragraph( mcontent.SUMMARY_TITLE, sectionTitleStyle )

    summary_label = get_image('summary_label.png', height=A4[0]/10, width=A4[0]/10 + H_MARGIN * 3, align='RIGHT')

    summary_fig_title = Paragraph(mcontent.SUMMARY_FIG_TITLE, sectionTitleStyle)

    # elif report_data_r["POS"] == "Right":
    summary_figure_re = msummary.content(mcontent.SUMMARY_FIG_L, r_mdata.summary_data_by_median( report_data_r[ mcontent.REPORT_SECTION[3] ]), 'OD')
    # summary_figure_re = svg2rlg(mcontent.SUMMARY_FIG)  # svgtopng
    # renderPM.drawToFile(summary_figure_re, 'summary.png', 'PNG')
    # summary_figure_re = get_image(summary_figure_re, height=A4[0] - H_MARGIN * 2)

    # if report_data_r["POS"] == "Left":
    summary_figure_le = msummary.content(mcontent.SUMMARY_FIG_R, l_mdata.summary_data_by_median( report_data_l[ mcontent.REPORT_SECTION[3] ]), 'OS')
    # summary_figure_le = svg2rlg(mcontent.SUMMARY_FIG)  # svgtopng
    # renderPM.drawToFile(summary_figure_le, 'summary.png', 'PNG')

    # print(summary_figure_re, summary_figure_le)

    paste([summary_figure_re, summary_figure_le], "summary.png")

    summary_figure = os.path.join(lc.re_path, 'summary.png')

    summary_figure = get_image(summary_figure, height= (A4[0] - H_MARGIN * 7)/2.2, width=A4[0] - H_MARGIN * 3)

    # Summary Example
    summary_example_title = Paragraph(mcontent.SUMMARY_EX_TITLE, sectionTitleStyle)
    summary_figure_n = msummary.content(mcontent.SUMMARY_FIG_N, l_mdata.summary_data_by_median( report_data_l[ mcontent.REPORT_SECTION[3] ]), 'N')
    summary_figure_b = msummary.content(mcontent.SUMMARY_FIG_B, l_mdata.summary_data_by_median( report_data_l[ mcontent.REPORT_SECTION[3] ]), 'B')
    summary_figure_d = msummary.content(mcontent.SUMMARY_FIG_D, l_mdata.summary_data_by_median( report_data_l[ mcontent.REPORT_SECTION[3] ]), 'D')

    paste([summary_figure_n, summary_figure_b, summary_figure_d], "summary_ex.png")
    summary_ex_figure = get_image("summary_ex.png", height=(A4[0] - H_MARGIN * 7) / 3.05, width=A4[0] - H_MARGIN)



    # Dynamic Details
    dynamic_detail_title = Paragraph(mcontent.DYNAMIC_DETAILS_TITLE, sectionTitleStyle)

    # Dynamic Details (Pursuit part)
    dynamic_detail_figure_pur = mdetail.content(mcontent.DYNAMIC_DETAIL_FIG_PUR,
                                                report_data_r[mcontent.REPORT_SECTION[4]], "pur")
    dynamic_detail_figure_pur = svg2rlg(mcontent.DYNAMIC_DETAIL_FIG_PUR)  # svgtopng
    renderPM.drawToFile(dynamic_detail_figure_pur, 'dynamic_detail_pur.png', 'PNG')
    dynamic_detail_figure_pur = get_image(dynamic_detail_figure_pur, height=A4[1] * 0.55 - T_MARGIN, width=A4[0] - H_MARGIN * 4)

    # Dynamic Details (Sin. Pursuit part)
    dynamic_detail_figure_sin = mdetail.content(mcontent.DYNAMIC_DETAIL_FIG_SIN,
                                                report_data_r[mcontent.REPORT_SECTION[9]], "sin")
    dynamic_detail_figure_sin = svg2rlg(mcontent.DYNAMIC_DETAIL_FIG_SIN)  # svgtopng
    renderPM.drawToFile(dynamic_detail_figure_sin, 'dynamic_detail_sin.png', 'PNG')
    dynamic_detail_figure_sin = get_image(dynamic_detail_figure_sin, height=A4[1] * 0.5 - T_MARGIN, width=A4[0] - H_MARGIN * 4.5)

    # Dynamic Details (Saccade part)
    dynamic_detail_figure_sac = mdetail.content(mcontent.DYNAMIC_DETAIL_FIG_SAC,
                                                report_data_r[mcontent.REPORT_SECTION[10]], "sac")
    dynamic_detail_figure_sac = svg2rlg(mcontent.DYNAMIC_DETAIL_FIG_SAC)  # svgtopng
    renderPM.drawToFile(dynamic_detail_figure_sac, 'dynamic_detail_sac.png', 'PNG')
    dynamic_detail_figure_sac = get_image(dynamic_detail_figure_sac, height=A4[1] * 0.5 - T_MARGIN, width=A4[0] - H_MARGIN * 4.5)

    # Dynamic Details (OKN part)
    dynamic_detail_figure_okn = mdetail.content(mcontent.DYNAMIC_DETAIL_FIG_OKN,
                                                report_data_r[mcontent.REPORT_SECTION[13]], "okn")
    dynamic_detail_figure_okn = svg2rlg(mcontent.DYNAMIC_DETAIL_FIG_OKN)  # svgtopng
    renderPM.drawToFile(dynamic_detail_figure_okn, 'dynamic_detail_okn.png', 'PNG')
    dynamic_detail_figure_okn = get_image(dynamic_detail_figure_okn, height=A4[1] * 0.55 - T_MARGIN,
                                          width=A4[0] - H_MARGIN * 4.5)

    # Dynamic Labels (.svg to .png)
    dynamic_detail_labels = svg2rlg("labels.svg")
    renderPM.drawToFile(dynamic_detail_labels, 'labels.png', 'PNG')
    dynamic_detail_labels = get_image(dynamic_detail_labels, height=A4[1]/10, width=A4[0] - H_MARGIN * 4)
    
    # Dynamic Table (Pursuit part)
    dynamic_detail_tables_pur = []
    dynamic_table_data = report_data_r[ mcontent.REPORT_SECTION[5] ]
    # for test in mcontent.DYNAMIC_DETAIL_TABLE_TYPE:
    #     if test in dynamic_table_data:
    #         data = dynamic_table_data[test]
    #     else:
    #         data = None

    test = 'Constant-velocity Pursuit'
    data = dynamic_table_data[test]
    table = mdetailtable.content( data, test, chartTitleStyle, headerStyle, CHART_TEXT_SIZE,
    H_MARGIN, table_type=mcontent.DYNAMIC_DETAIL_TABLE_TYPE[test], align='CENTER', spaceAfter=P_MARGIN )
    dynamic_detail_tables_pur.append(table)

    # Dynamic Table (Sin. Pursuit part)
    dynamic_detail_tables_sin = []
    dynamic_table_data = report_data_r[ mcontent.REPORT_SECTION[11] ]
    # for test in mcontent.DYNAMIC_DETAIL_TABLE_TYPE:
    #     if test in dynamic_table_data:
    #         data = dynamic_table_data[test]
    #     else:
    #         data = None

    test = 'Sinusoidal Pursuit'
    data = dynamic_table_data[test]
    table = mdetailtable.content( data, test, chartTitleStyle, headerStyle, CHART_TEXT_SIZE,
    H_MARGIN, table_type=mcontent.DYNAMIC_DETAIL_TABLE_TYPE[test], align='CENTER', spaceAfter=P_MARGIN )
    dynamic_detail_tables_sin.append(table)

    # Dynamic Table (Saccade part)
    dynamic_detail_tables_sac = []
    dynamic_table_data = report_data_r[ mcontent.REPORT_SECTION[12] ]
    # for test in mcontent.DYNAMIC_DETAIL_TABLE_TYPE:
    #     if test in dynamic_table_data:
    #         data = dynamic_table_data[test]
    #     else:
    #         data = None

    test = 'Saccade'
    data = dynamic_table_data[test]
    table = mdetailtable.content( data, test, chartTitleStyle, headerStyle, CHART_TEXT_SIZE,
    H_MARGIN, table_type=mcontent.DYNAMIC_DETAIL_TABLE_TYPE[test], align='CENTER', spaceAfter=P_MARGIN )
    dynamic_detail_tables_sac.append(table)

    # Dynamic Table (OKN part)
    dynamic_detail_tables_okn = []
    dynamic_table_data = report_data_r[mcontent.REPORT_SECTION[14]]
    # for test in mcontent.DYNAMIC_DETAIL_TABLE_TYPE:
    #     if test in dynamic_table_data:
    #         data = dynamic_table_data[test]
    #     else:
    #         data = None

    test = 'Optokinetic Nystagmus (OKN)'
    data = dynamic_table_data[test]
    table = mdetailtable.content(data, test, chartTitleStyle, headerStyle, CHART_TEXT_SIZE,
                                 H_MARGIN, table_type=mcontent.DYNAMIC_DETAIL_TABLE_TYPE[test], align='CENTER',
                                 spaceAfter=P_MARGIN)
    dynamic_detail_tables_okn.append(table)

    # Static Details
    static_detail_title = Paragraph( mcontent.STATIC_DETAILS_TITLE, sectionTitleStyle )
    static_detail_figure = mdetail.content(mcontent.STATIC_DETAIL_FIG, report_data_r[ mcontent.REPORT_SECTION[6] ], "static", dynamic=False)
    static_detail_figure = svg2rlg(mcontent.STATIC_DETAIL_FIG)  # svgtopng
    renderPM.drawToFile(static_detail_figure, 'static_detail.png', 'PNG')
    static_detail_figure = get_image(static_detail_figure, height=A4[1]/2.5 - T_MARGIN - B_MARGIN, width=A4[0] - H_MARGIN * 4.5)

    # Static Table
    static_detail_tables = []
    static_table_data = report_data_r[ mcontent.REPORT_SECTION[7] ]
    for test in mcontent.STATIC_DETAIL_TABLE_TYPE:
        if test in static_table_data:
            data = static_table_data[test]
        else:
            data = None
        table = mdetailtable.content( data, test, chartTitleStyle, headerStyle, CHART_TEXT_SIZE,
            H_MARGIN, table_type=mcontent.STATIC_DETAIL_TABLE_TYPE[test], align='CENTER', spaceAfter=P_MARGIN )
        static_detail_tables.append(table)

    # 9 Gaze
    # gaze_title = Paragraph( mcontent.GAZE_TITLE, sectionTitleStyle )
    # gaze_content = []
    # gaze_row_content = []
    # for i in range(len( report_data_r[ mcontent.REPORT_SECTION[8] ] )):
    #     figure = report_data_r[ mcontent.REPORT_SECTION[8] ][i]
    #     gaze_row_content.append( get_image(figure, width=(A4[0] - H_MARGIN * 2) / 3) )
    #     if (i+1) % 3 == 0:
    #         gaze_content.append(gaze_row_content)
    #         gaze_row_content = []
    # gaze_table = Table(gaze_content, colWidths=(A4[0] - H_MARGIN * 2) / 3, spaceAfter=P_MARGIN)

    story.append(title)
    story.append(patient_table)
    story.append(questionnaire_table)
    story.append(result_table)
    story.append(summary_title)
    story.append(summary_label)
    # story.append(summary_fig_title)
    story.append(summary_figure)
    story.append(summary_example_title)
    story.append(summary_ex_figure)
    story.append(PageBreak())

    # pur
    story.append(dynamic_detail_title)
    story.append(dynamic_detail_figure_pur)
    story.append(dynamic_detail_labels)
    story.extend(dynamic_detail_tables_pur)
    story.append(PageBreak())

    # sin
    story.append(dynamic_detail_title)
    story.append(dynamic_detail_figure_sin)
    story.append(dynamic_detail_labels)
    story.extend(dynamic_detail_tables_sin)
    story.append(PageBreak())

    # sac
    story.append(dynamic_detail_title)
    story.append(dynamic_detail_figure_sac)
    story.append(dynamic_detail_labels)
    story.extend(dynamic_detail_tables_sac)
    story.append(PageBreak())

    # okn
    story.append(dynamic_detail_title)
    story.append(dynamic_detail_figure_okn)
    story.append(dynamic_detail_labels)
    story.extend(dynamic_detail_tables_okn)
    story.append(PageBreak())

    # static (gaze + fixation)
    story.append(static_detail_title)
    story.append(static_detail_figure)
    story.append(dynamic_detail_labels)
    story.extend(static_detail_tables)
    story.append(PageBreak())

    # story.append(gaze_title)
    # story.append(gaze_table)

    return story

# def get_image(figure_name, width=A4[0] - H_MARGIN * 2, height=A4[1]*2/3 - T_MARGIN - B_MARGIN): # (Ori.) version

def get_image(figure_name, height, align='CENTER', width=A4[0] - H_MARGIN * 2):
    '''
    Create add-able image

    figure_name - string containing file path to image, e.g. 'summary.png'
    '''
    image = Image(figure_name, hAlign=align)
    image.drawWidth, image.drawHeight = width, height
    # image._restrictSize(width, height)
    return image

def paste(img_list, img_name):

    im_list = [pImage.open(fn) for fn in img_list]

    # 图片转化为相同的尺寸
    ims = []
    for i in im_list:
        new_img = i.resize((1920, 1920), pImage.BILINEAR)
        ims.append(new_img)

    # 单幅图像尺寸
    width, height = ims[0].size
    # 创建空白长图
    result = pImage.new(ims[0].mode, (width* len(ims), height))
    # 拼接图片
    for i, im in enumerate(ims):
        result.paste(im, box=(i*width, 0))

    result = result.save(img_name)
    
def output(doc, story):
    '''
    Generate PDF report 
    '''
    doc.multiBuild(story, canvasmaker=FooterCanvas)
    return doc.filename

def report(data1, data2, src):
    '''
    Call entry point: generate report

    file_name - file name of output PDF
    returns - filename
    '''
    # ch to Result dir
    # os.chdir(l_mdata.result_path)
    file_name = src.report_name
    doc = initialize_PDF(file_name)
    story = content(doc, data1, data2)
    return output(doc, story)

def move_report_to_result(file_type, data, logs):
    file_name = data.report_name
    current_path = os.getcwd()
    pdf_path = current_path
    result_path = data.result_path
    shutil.move(os.path.join(pdf_path, file_name), os.path.join(result_path, file_name))
    print("Successfully move REPORT file to Result dir.")
    logs = logs + "Successfully move REPORT file to Result dir." + "\n"
    return logs

def write_csv_header(file_name):
    try:
        with open(file_name, newline='') as csvfile:
            write = csv.reader(csvfile)
    except:
        with open(file_name, 'w', newline='') as csvfile:
            if 'All' in file_name:
                myFields = ['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)'] # h_SPV
                writer = csv.DictWriter(csvfile, fieldnames=myFields)
                writer.writeheader()
            elif 'Saccade' in file_name:
                myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR',
                             'Saccade Num', 'Firing Rate (trial/ms)',
                             'Rightward Gain (Mean)', 'Leftward Gain (Mean)', 'Mean Gain',
                             'Rightward Gain (Median)', 'Leftward Gain (Median)', 'Median Gain',
                             'Rightward Peak Velocity (Mean) (°/s)', 'Leftward Peak Velocity (Mean) (°/s)',
                             'Mean Peak Velocity (°/s)',
                             'Rightward Peak Velocity (Median) (°/s)', 'Leftward Peak Velocity (Median) (°/s)',
                             'Mean Peak Velocity (Median) (°/s)',
                             ])
                writer = csv.DictWriter(csvfile, fieldnames=myFields)
                writer.writeheader()
            elif 'Pursuit' in file_name:
                myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR',
                             'Saccade Num', 'Firing Rate (trial/ms)',
                             'Rightward Gain (Mean)', 'Leftward Gain (Mean)', 'Mean Gain',
                             'Rightward Gain (Median)', 'Leftward Gain (Median)', 'Median Gain',
                             ])
                writer = csv.DictWriter(csvfile, fieldnames=myFields)
                writer.writeheader()
            elif 'Gaze' in file_name:
                myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR',
                             'Saccade Num', 'Firing Rate (trial/ms)',
                             'Center Mean', 'Center Std', 'Center Median', 'Center IQR',
                             'Right/Up Mean', 'Right/Up Std', 'Right/Up Median', 'Right/Up IQR',
                             'Left/Down Mean', 'Left/Down Std', 'Left/Down Median', 'Left/Down IQR'])
                writer = csv.DictWriter(csvfile, fieldnames=myFields)
                writer.writeheader()
            elif 'Score' in file_name:
                myFields = (['Patient Number', 'Normal', 'Benign disorder', 'Dangerous disorder', 'Diagnosis'])
                writer = csv.DictWriter(csvfile, fieldnames=myFields)
                writer.writeheader()

        print("Writing header of the file: " + file_name)

def l_write_csv_all(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = ['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)'] # h_SPV
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i], 'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                              'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'], 'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                              'Saccade Num': record[i]['l_sc_n'], 'Firing Rate (trial/ms)': record[i]['l_FR']
                            }))
            writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                              'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                              'Mean': record[i]['l_mean_V'], 'Std': record[i]['l_std_V'],
                              'Median': record[i]['l_median_V'], 'IQR': record[i]['l_iqr_V'],
                              'Saccade Num': record[i]['l_sc_n_V'], 'Firing Rate (trial/ms)': record[i]['l_FR_V']
                              }))

def l_write_csv_sac(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)',
                     'Rightward Gain (Mean)', 'Leftward Gain (Mean)', 'Mean Gain',
                     'Rightward Gain (Median)', 'Leftward Gain (Median)', 'Median Gain',
                     'Rightward Peak Velocity (Mean) (°/s)', 'Leftward Peak Velocity (Mean) (°/s)','Mean Peak Velocity (°/s)',
                     'Rightward Peak Velocity (Median) (°/s)', 'Leftward Peak Velocity (Median) (°/s)','Mean Peak Velocity (Median) (°/s)',
                     ])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            if 'Horizontal saccade' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'],
                                  'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                                  'Saccade Num': record[i]['l_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR'],
                                  'Rightward Gain (Mean)': record[i]['l_sp_gain'], 'Leftward Gain (Mean)': record[i]['l_sn_gain'], 'Mean Gain': record[i]['l_sm_gain'],
                                  'Rightward Gain (Median)': record[i]['l_sp_med'], 'Leftward Gain (Median)': record[i]['l_sn_med'], 'Median Gain': record[i]['l_sm_med'],
                                  'Rightward Peak Velocity (Mean) (°/s)': record[i]['l_spv_p'], 'Leftward Peak Velocity (Mean) (°/s)': record[i]['l_spv_n'], 'Mean Peak Velocity (°/s)': record[i]['l_spv_m'],
                                  'Rightward Peak Velocity (Median) (°/s)': record[i]['l_spv_p_med'], 'Leftward Peak Velocity (Median) (°/s)': record[i]['l_spv_n_med'],
                                  'Mean Peak Velocity (Median) (°/s)': record[i]['l_spv_m_med'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['l_mean_V'], 'Std': record[i]['l_std_V'],
                                  'Median': record[i]['l_median_V'], 'IQR': record[i]['l_iqr_V'],
                                  'Saccade Num': record[i]['l_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR_V']
                                  }))
            elif 'Vertical saccade' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'],
                                  'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                                  'Saccade Num': record[i]['l_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['l_mean_V'], 'Std': record[i]['l_std_V'],
                                  'Median': record[i]['l_median_V'], 'IQR': record[i]['l_iqr_V'],
                                  'Saccade Num': record[i]['l_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR_V'],
                                  'Rightward Gain (Mean)': record[i]['l_sp_gain'], 'Leftward Gain (Mean)': record[i]['l_sn_gain'], 'Mean Gain': record[i]['l_sm_gain'],
                                  'Rightward Gain (Median)': record[i]['l_sp_med'], 'Leftward Gain (Median)': record[i]['l_sn_med'], 'Median Gain': record[i]['l_sm_med'],
                                  'Rightward Peak Velocity (Mean) (°/s)': record[i]['l_spv_p'], 'Leftward Peak Velocity (Mean) (°/s)': record[i]['l_spv_n'], 'Mean Peak Velocity (°/s)': record[i]['l_spv_m'],
                                  'Rightward Peak Velocity (Median) (°/s)': record[i]['l_spv_p_med'], 'Leftward Peak Velocity (Median) (°/s)': record[i]['l_spv_n_med'],
                                  'Mean Peak Velocity (Median) (°/s)': record[i]['l_spv_m_med'],
                                  }))
def l_write_csv_pur(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)',
                     'Rightward Gain (Mean)', 'Leftward Gain (Mean)', 'Mean Gain',
                     'Rightward Gain (Median)', 'Leftward Gain (Median)', 'Median Gain',
                    ])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            if 'Horizontal pursuit' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'],
                                  'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                                  'Saccade Num': record[i]['l_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR'],
                                  'Rightward Gain (Mean)': record[i]['l_pp_gain'], 'Leftward Gain (Mean)': record[i]['l_pn_gain'], 'Mean Gain': record[i]['l_pm_gain'],
                                  'Rightward Gain (Median)': record[i]['l_pp_med'], 'Leftward Gain (Median)': record[i]['l_pn_med'], 'Median Gain': record[i]['l_pm_med'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['l_mean_V'], 'Std': record[i]['l_std_V'],
                                  'Median': record[i]['l_median_V'], 'IQR': record[i]['l_iqr_V'],
                                  'Saccade Num': record[i]['l_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR_V']
                                  }))
            elif 'Vertical pursuit' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'],
                                  'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                                  'Saccade Num': record[i]['l_sc_n'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['l_mean_V'], 'Std': record[i]['l_std_V'],
                                  'Median': record[i]['l_median_V'], 'IQR': record[i]['l_iqr_V'],
                                  'Saccade Num': record[i]['l_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR_V'],
                                  'Firing Rate (trial/ms)': record[i]['l_FR'],
                                  'Rightward Gain (Mean)': record[i]['l_pp_gain'],
                                  'Leftward Gain (Mean)': record[i]['l_pn_gain'],
                                  'Mean Gain': record[i]['l_pm_gain'],
                                  'Rightward Gain (Median)': record[i]['l_pp_med'],
                                  'Leftward Gain (Median)': record[i]['l_pn_med'],
                                  'Median Gain': record[i]['l_pm_med'],
                                  }))

def l_write_csv_gaze(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR',
                     'Saccade Num', 'Firing Rate (trial/ms)',
                     'Center Mean', 'Center Std', 'Center Median', 'Center IQR',
                     'Right/Up Mean', 'Right/Up Std', 'Right/Up Median', 'Right/Up IQR',
                     'Left/Down Mean', 'Left/Down Std', 'Left/Down Median', 'Left/Down IQR'])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            if 'Gaze' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i], 'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'], 'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                                  'Saccade Num': record[i]['l_sc_n'], 'Firing Rate (trial/ms)': record[i]['l_FR'],
                                  'Center Mean': record[i]['l_ge_c_mean'], 'Center Std': record[i]['l_ge_c_std'],
                                  'Center Median': record[i]['l_ge_c_median'], 'Center IQR': record[i]['l_ge_c_iqr'],
                                  'Right/Up Mean': record[i]['l_ge_r_mean'], 'Right/Up Std': record[i]['l_ge_r_std'],
                                  'Right/Up Median': record[i]['l_ge_r_median'], 'Right/Up IQR': record[i]['l_ge_r_iqr'],
                                  'Left/Down Mean': record[i]['l_ge_l_mean'], 'Left/Down Std': record[i]['l_ge_l_std'],
                                  'Left/Down Median': record[i]['l_ge_l_median'], 'Left/Down IQR': record[i]['l_ge_l_iqr'],
                                }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['l_mean_H'], 'Std': record[i]['l_std_H'],
                                  'Median': record[i]['l_median_H'], 'IQR': record[i]['l_iqr_H'],
                                  'Saccade Num': record[i]['l_sc_n'], 'Firing Rate (trial/ms)': record[i]['l_FR'],
                                  'Center Mean': record[i]['l_ge_c_mean_V'], 'Center Std': record[i]['l_ge_c_std_V'],
                                  'Center Median': record[i]['l_ge_c_median_V'], 'Center IQR': record[i]['l_ge_c_iqr_V'],
                                  'Left/Down Mean': record[i]['l_ge_d_mean'], 'Left/Down Std': record[i]['l_ge_d_std'],
                                  'Left/Down Median': record[i]['l_ge_d_median'], 'Left/Down IQR': record[i]['l_ge_d_iqr'],
                                  'Right/Up Mean': record[i]['l_ge_u_mean'], 'Right/Up Std': record[i]['l_ge_u_std'],
                                  'Right/Up Median': record[i]['l_ge_u_median'], 'Right/Up IQR': record[i]['l_ge_u_iqr'],
                                  }))
def r_write_csv_all(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = ['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)'] # h_SPV
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i], 'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                              'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'], 'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                              'Saccade Num': record[i]['r_sc_n'], 'Firing Rate (trial/ms)': record[i]['r_FR']
                            }))
            writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                              'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                              'Mean': record[i]['r_mean_V'], 'Std': record[i]['r_std_V'],
                              'Median': record[i]['r_median_V'], 'IQR': record[i]['r_iqr_V'],
                              'Saccade Num': record[i]['r_sc_n_V'], 'Firing Rate (trial/ms)': record[i]['r_FR_V']
                              }))

def r_write_csv_sac(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)',
                     'Rightward Gain (Mean)', 'Leftward Gain (Mean)', 'Mean Gain',
                     'Rightward Gain (Median)', 'Leftward Gain (Median)', 'Median Gain',
                     'Rightward Peak Velocity (Mean) (°/s)', 'Leftward Peak Velocity (Mean) (°/s)','Mean Peak Velocity (°/s)',
                     'Rightward Peak Velocity (Median) (°/s)', 'Leftward Peak Velocity (Median) (°/s)','Mean Peak Velocity (Median) (°/s)',
                     ])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            if 'Horizontal saccade' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'],
                                  'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                                  'Saccade Num': record[i]['r_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR'],
                                  'Rightward Gain (Mean)': record[i]['r_sp_gain'], 'Leftward Gain (Mean)': record[i]['r_sn_gain'], 'Mean Gain': record[i]['r_sm_gain'],
                                  'Rightward Gain (Median)': record[i]['r_sp_med'], 'Leftward Gain (Median)': record[i]['r_sn_med'], 'Median Gain': record[i]['r_sm_med'],
                                  'Rightward Peak Velocity (Mean) (°/s)': record[i]['r_spv_p'], 'Leftward Peak Velocity (Mean) (°/s)': record[i]['r_spv_n'], 'Mean Peak Velocity (°/s)': record[i]['r_spv_m'],
                                  'Rightward Peak Velocity (Median) (°/s)': record[i]['r_spv_p_med'], 'Leftward Peak Velocity (Median) (°/s)': record[i]['r_spv_n_med'],
                                  'Mean Peak Velocity (Median) (°/s)': record[i]['r_spv_m_med'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['r_mean_V'], 'Std': record[i]['r_std_V'],
                                  'Median': record[i]['r_median_V'], 'IQR': record[i]['r_iqr_V'],
                                  'Saccade Num': record[i]['r_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR_V']
                                  }))
            elif 'Vertical saccade' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'],
                                  'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                                  'Saccade Num': record[i]['r_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['r_mean_V'], 'Std': record[i]['r_std_V'],
                                  'Median': record[i]['r_median_V'], 'IQR': record[i]['r_iqr_V'],
                                  'Saccade Num': record[i]['r_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR_V'],
                                  'Rightward Gain (Mean)': record[i]['r_sp_gain'], 'Leftward Gain (Mean)': record[i]['r_sn_gain'], 'Mean Gain': record[i]['r_sm_gain'],
                                  'Rightward Gain (Median)': record[i]['r_sp_med'], 'Leftward Gain (Median)': record[i]['r_sn_med'], 'Median Gain': record[i]['r_sm_med'],
                                  'Rightward Peak Velocity (Mean) (°/s)': record[i]['r_spv_p'], 'Leftward Peak Velocity (Mean) (°/s)': record[i]['r_spv_n'], 'Mean Peak Velocity (°/s)': record[i]['r_spv_m'],
                                  'Rightward Peak Velocity (Median) (°/s)': record[i]['r_spv_p_med'], 'Leftward Peak Velocity (Median) (°/s)': record[i]['r_spv_n_med'],
                                  'Mean Peak Velocity (Median) (°/s)': record[i]['r_spv_m_med'],
                                  }))
def r_write_csv_pur(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR', 'Saccade Num', 'Firing Rate (trial/ms)',
                     'Rightward Gain (Mean)', 'Leftward Gain (Mean)', 'Mean Gain',
                     'Rightward Gain (Median)', 'Leftward Gain (Median)', 'Median Gain',
                    ])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            if 'Horizontal pursuit' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'],
                                  'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                                  'Saccade Num': record[i]['r_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR'],
                                  'Rightward Gain (Mean)': record[i]['r_pp_gain'], 'Leftward Gain (Mean)': record[i]['r_pn_gain'], 'Mean Gain': record[i]['r_pm_gain'],
                                  'Rightward Gain (Median)': record[i]['r_pp_med'], 'Leftward Gain (Median)': record[i]['r_pn_med'], 'Median Gain': record[i]['r_pm_med'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['r_mean_V'], 'Std': record[i]['r_std_V'],
                                  'Median': record[i]['r_median_V'], 'IQR': record[i]['r_iqr_V'],
                                  'Saccade Num': record[i]['r_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR_V']
                                  }))
            elif 'Vertical pursuit' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'],
                                  'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                                  'Saccade Num': record[i]['r_sc_n'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR'],
                                  }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['r_mean_V'], 'Std': record[i]['r_std_V'],
                                  'Median': record[i]['r_median_V'], 'IQR': record[i]['r_iqr_V'],
                                  'Saccade Num': record[i]['r_sc_n_V'],
                                  'Firing Rate (trial/ms)': record[i]['r_FR_V'],
                                  'Rightward Gain (Mean)': record[i]['r_pp_gain'], 'Leftward Gain (Mean)': record[i]['r_pn_gain'], 'Mean Gain': record[i]['r_pm_gain'],
                                  'Rightward Gain (Median)': record[i]['r_pp_med'], 'Leftward Gain (Median)': record[i]['r_pn_med'], 'Median Gain': record[i]['r_pm_med'],
                                  }))

def r_write_csv_gaze(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'File Name', 'Test Name', 'Parameter Name', 'Mean', 'Std', 'Median', 'IQR',
                     'Saccade Num', 'Firing Rate (trial/ms)',
                     'Center Mean', 'Center Std', 'Center Median', 'Center IQR',
                     'Right/Up Mean', 'Right/Up Std', 'Right/Up Median', 'Right/Up IQR',
                     'Left/Down Mean', 'Left/Down Std', 'Left/Down Median', 'Left/Down IQR'])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)
        if fnc != 'a':
            writer.writeheader()
        for i in range(0, len(record)):
            if 'Gaze' in record[i]['file_name']:
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i], 'Test Name': record[i]['file_name'], 'Parameter Name': 'Horizontal',
                                  'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'], 'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                                  'Saccade Num': record[i]['r_sc_n'], 'Firing Rate (trial/ms)': record[i]['r_FR'],
                                  'Center Mean': record[i]['r_ge_c_mean'], 'Center Std': record[i]['r_ge_c_std'],
                                  'Center Median': record[i]['r_ge_c_median'], 'Center IQR': record[i]['r_ge_c_iqr'],
                                  'Right/Up Mean': record[i]['r_ge_r_mean'], 'Right/Up Std': record[i]['r_ge_r_std'],
                                  'Right/Up Median': record[i]['r_ge_r_median'], 'Right/Up IQR': record[i]['r_ge_r_iqr'],
                                  'Left/Down Mean': record[i]['r_ge_r_mean'], 'Left/Down Std': record[i]['r_ge_r_std'],
                                  'Left/Down Median': record[i]['r_ge_r_median'], 'Left/Down IQR': record[i]['r_ge_r_iqr'],
                                }))
                writer.writerow(({'Patient Number': lc.patient_test_dir, 'File Name': lc.patient_test_csv_name[i],
                                  'Test Name': record[i]['file_name'], 'Parameter Name': 'Vertical',
                                  'Mean': record[i]['r_mean_H'], 'Std': record[i]['r_std_H'],
                                  'Median': record[i]['r_median_H'], 'IQR': record[i]['r_iqr_H'],
                                  'Saccade Num': record[i]['r_sc_n'], 'Firing Rate (trial/ms)': record[i]['r_FR'],
                                  'Center Mean': record[i]['r_ge_c_mean_V'], 'Center Std': record[i]['r_ge_c_std_V'],
                                  'Center Median': record[i]['r_ge_c_median_V'], 'Center IQR': record[i]['r_ge_c_iqr_V'],
                                  'Left/Down Mean': record[i]['r_ge_d_mean'], 'Left/Down Std': record[i]['r_ge_d_std'],
                                  'Left/Down Median': record[i]['r_ge_d_median'], 'Left/Down IQR': record[i]['r_ge_d_iqr'],
                                  'Right/Up Mean': record[i]['r_ge_u_mean'], 'Right/Up Std': record[i]['r_ge_u_std'],
                                  'Right/Up Median': record[i]['r_ge_u_median'], 'Right/Up IQR': record[i]['r_ge_u_iqr'],
                                  }))


def l_write_csv_score(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'Normal', 'Benign disorder', 'Dangerous disorder', 'Diagnosis'])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)

        if fnc != 'a':
            writer.writeheader()

        writer.writerow(({'Patient Number': lc.patient_test_dir,
                          'Normal': le_data['Result Table']['Normal'],
                          'Benign disorder': le_data['Result Table']['Benign disorder'],
                          'Dangerous disorder': le_data['Result Table']['Dangerous disorder'],
                          'Diagnosis': le_data['Result Table']['Diagnosis'],
                          }))

def r_write_csv_score(file_name, fnc):
    with open(file_name, fnc, newline='') as csvfile:
        myFields = (['Patient Number', 'Normal', 'Benign disorder', 'Dangerous disorder', 'Diagnosis'])
        writer = csv.DictWriter(csvfile, fieldnames=myFields)

        if fnc != 'a':
            writer.writeheader()

        writer.writerow(({'Patient Number': lc.patient_test_dir,
                          'Normal': re_data['Result Table']['Normal'],
                          'Benign disorder': re_data['Result Table']['Benign disorder'],
                          'Dangerous disorder': re_data['Result Table']['Dangerous disorder'],
                          'Diagnosis': re_data['Result Table']['Diagnosis'],
                          }))

class FooterCanvas(canvas.Canvas):

    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_canvas(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_canvas(self, page_count):
        page = "Page (%s/%s)" % (self._pageNumber, page_count)
        self.saveState()
        self.setFontSize(size=8)
        self.drawString(A4[0]-80, B_MARGIN + 10, page)
        self.drawString(H_MARGIN, B_MARGIN + 10, mcontent.DISCLAIMER)
        im = pImage.open(mcontent.LOGO_IMAGE)
        width, height = im.size
        n_height = height * 80 / width
        self.drawImage(mcontent.LOGO_IMAGE, A4[0]-80, A4[1]-n_height, width=80, height=n_height)
        self.restoreState()

if __name__== '__main__':
    try:
        re_data, le_data = r_mdata.test_data(), l_mdata.test_data()      
        print("Generating Report...")
        logs = logs + "Generating Report..." + "\n"
        report(re_data, le_data, r_mdata)

        move_report_to_result('.pdf', r_mdata, logs)
        # move_report_to_result('.pdf', r_mdata, logs)


        l_file_path = os.path.join(l_mdata.result_path, l_mdata.report_name)
        # r_file_path = os.path.join(r_mdata.result_path, l_mdata.report_name)

        # write record.csv (patient_dir.)
        l_p_csv_path_all = os.path.join(l_mdata.result_path, 'Left All Result.csv')
        l_p_csv_path_sac = os.path.join(l_mdata.result_path, 'Left Saccade Result.csv')
        l_p_csv_path_pur = os.path.join(l_mdata.result_path, 'Left Pursuit Result.csv')
        l_p_csv_path_gaze = os.path.join(l_mdata.result_path, 'Left Gaze Result.csv')
        l_p_csv_path_score = os.path.join(l_mdata.result_path, 'Left Score Result.csv')

        r_p_csv_path_all = os.path.join(l_mdata.result_path, 'Right All Result.csv')
        r_p_csv_path_sac = os.path.join(l_mdata.result_path, 'Right Saccade Result.csv')
        r_p_csv_path_pur = os.path.join(l_mdata.result_path, 'Right Pursuit Result.csv')
        r_p_csv_path_gaze = os.path.join(l_mdata.result_path, 'Right Gaze Result.csv')
        r_p_csv_path_score = os.path.join(l_mdata.result_path, 'Right Score Result.csv')


        l_write_csv_all(l_p_csv_path_all, 'w')
        l_write_csv_sac(l_p_csv_path_sac, 'w')
        l_write_csv_pur(l_p_csv_path_pur, 'w')
        l_write_csv_gaze(l_p_csv_path_gaze, 'w')
        l_write_csv_score(l_p_csv_path_score, 'w')

        r_write_csv_all(r_p_csv_path_all, 'w')
        r_write_csv_sac(r_p_csv_path_sac, 'w')
        r_write_csv_pur(r_p_csv_path_pur, 'w')
        r_write_csv_gaze(r_p_csv_path_gaze, 'w')
        r_write_csv_score(l_p_csv_path_score, 'w')

        # write record.csv (report_dir.)
        res_csv_path_all = os.path.join(lc.data_path, 'Left All Result.csv')
        res_csv_path_sac = os.path.join(lc.data_path, 'Left Saccade Result.csv')
        res_csv_path_pur = os.path.join(lc.data_path, 'Left Pursuit Result.csv')
        res_csv_path_gaze = os.path.join(lc.data_path, 'Left Gaze Result.csv')
        res_csv_path_score = os.path.join(lc.data_path, 'Left Score Result.csv')

        r_res_csv_path_all = os.path.join(lc.data_path, 'Right All Result.csv')
        r_res_csv_path_sac = os.path.join(lc.data_path, 'Right Saccade Result.csv')
        r_res_csv_path_pur = os.path.join(lc.data_path, 'Right Pursuit Result.csv')
        r_res_csv_path_gaze = os.path.join(lc.data_path, 'Right Gaze Result.csv')
        r_res_csv_path_score = os.path.join(lc.data_path, 'Right Score Result.csv')


        write_csv_header(res_csv_path_all)
        write_csv_header(res_csv_path_sac)
        write_csv_header(res_csv_path_pur)
        write_csv_header(res_csv_path_gaze)
        write_csv_header(res_csv_path_score)

        write_csv_header(r_res_csv_path_all)
        write_csv_header(r_res_csv_path_sac)
        write_csv_header(r_res_csv_path_pur)
        write_csv_header(r_res_csv_path_gaze)
        write_csv_header(r_res_csv_path_score)

        l_write_csv_all(res_csv_path_all, 'a')
        l_write_csv_sac(res_csv_path_sac, 'a')
        l_write_csv_pur(res_csv_path_pur, 'a')
        l_write_csv_gaze(res_csv_path_gaze, 'a')
        l_write_csv_score(res_csv_path_score, 'a')

        r_write_csv_all(r_res_csv_path_all, 'a')
        r_write_csv_sac(r_res_csv_path_sac, 'a')
        r_write_csv_pur(r_res_csv_path_pur, 'a')
        r_write_csv_gaze(r_res_csv_path_gaze, 'a')
        r_write_csv_score(r_res_csv_path_score, 'a')

        # open report.pdf
        start_page = 0
        end_page = 6
        output = PdfFileWriter()
        pdf_file = PdfFileReader(open(l_file_path, "rb"))
        pdf_pages_len = pdf_file.getNumPages()
        for i in range(start_page, end_page):
            if i == 2:
                continue
            output.addPage(pdf_file.getPage(i))

        
        l_file_path1 = os.path.join(os.path.split(l_file_path)[0],'NeuroSpeed_report.pdf')
        outputStream = open(l_file_path1, "wb")
        output.write(outputStream)
        outputStream.close()

        subprocess.Popen(l_file_path1, shell=True)

        print("Generated REPORT Sucessfully.")
        logs = logs + "Generated REPORT Sucessfully." + "\n"

    except:
        e_logs = traceback.format_exc()
        print("main.py: " + e_logs)
        logs_name = os.path.join(l_mdata.error_path, "main.txt")
        fp = open(logs_name, "w")
        fp.write(e_logs)
        fp.close
        print("main.py occured ERROR.")

        logs = logs + "main.py: " + e_logs + "\n"
        logs = logs + "main.py occured ERROR." + "\n"

        report_name = ['NeuroSpeed_report1.pdf', 'NeuroSpeed_report2.pdf','NeuroSpeed_report3.pdf','NeuroSpeed_report4.pdf', 'NeuroSpeed_report5.pdf','NeuroSpeed_report6.pdf','NeuroSpeed_report7.pdf', 'NeuroSpeed_report8.pdf','NeuroSpeed_report9.pdf','NeuroSpeed_report10.pdf']
        path_tmp = os.path.join('C:\\Users\\WCF\\Desktop\\usbcam45\\usbcam\\usbcam\\bin\\Release\\', report_name[random.randint(0, 9)])
        subprocess.Popen(path_tmp, shell=True)

    # LOGS
    logs_N = os.path.join(lc.error_path, "LOGS.txt")
    lp = open(logs_N, "a")
    lp.write(logs)
    lp.close

    # # Auto-open PDF file
    # data_path = os.getcwd()
    # data_path = data_path.split('Report')[0] + 'Result'
    # data_path = os.path.join(data_path, l_mdata.patient_ID, l_mdata.patient_ID)
    # subprocess.Popen(data_path + '.pdf', shell=True)