import content as mcontent
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

def content(data, table_title, title_style, header_style, text_size, page_margin, table_type , align='LEFT', spaceAfter=10):
    '''
    data - pass me data!
    table_title - title of table
    table_type - 'okn' | 'pursuit' | 'saccade'
    returns - Table
    '''
    content = []

    if table_type == 'okn':
        table = make_content(data, table_title, title_style, header_style, text_size, page_margin, mcontent.OKN_DATA, mcontent.OKN_TYPE)
    elif table_type == 'saccade':
        table = make_content(data, table_title, title_style, header_style, text_size, page_margin, mcontent.SACCADE_DATA, mcontent.SACCADE_TYPE)
    elif table_type == 'pursuit':
        table = make_content(data, table_title, title_style, header_style, text_size, page_margin, mcontent.PURSUIT_DATA, mcontent.PURSUIT_TYPE)
    elif table_type == 'sin-pursuit':
        table = make_content(data, table_title, title_style, header_style, text_size, page_margin, mcontent.PURSUIT_DATA, mcontent.SIN_PURSUIT_TYPE)
    elif table_type == 'gaze': # Gaze 10s -> Gaze
        table = make_content(data, table_title.split(" ")[0], title_style, header_style, 6, page_margin, mcontent.GAZE_DATA, mcontent.GAZE_TYPE)
    elif table_type == 'fix':
        table = make_content(data, table_title, title_style, header_style, text_size, page_margin, mcontent.FIX_DATA, mcontent.FIX_TYPE)
    else:
        table = Table([table_title])

    return table

def make_content(data, table_title, title_style, header_style, text_size, page_margin, data_cat, type_cat, align='CENTER', spaceAfter=10):
    content = []
    content.append( [ Paragraph('<b><i>%s</i></b>' % table_title, title_style)] )

    headers = ['', '']
    content_spaces = []
    for t in type_cat:
        if t == 'Positive':
            if 'OKN' in table_title:
                t = table_title + '(+)'
            elif 'Fixation' in table_title:
                t = ''
        elif t == 'Negative':
            if 'OKN' in table_title:
                t = table_title + '(-)'
            elif 'Fixation' in table_title:
                t = ''
        headers.append(Paragraph(t, header_style))
        content_spaces.append('')
    content.append(headers)

    for k in data_cat:
        for d in data_cat[k]:
            if d == 'Mean':
                d = d + mcontent.RANGE_STD
            elif d == 'Median':
                d = d + mcontent.RANGE_IQR
            elif d == 'Duration':
                d = d + mcontent.UNIT_S
            elif d == 'Peak Velocity' or d == 'Amplitude':
                d = d + mcontent.UNIT_DEGREE
            data_content = [k, d]
            data_content.extend(content_spaces)
            content.append( data_content )

    for t in range(len(type_cat)):
        key = type_cat[t]
        j = t + 2
        if not data is None and key in data:
            type_data = data[key]
            row = 2
            for k in data_cat:
                if k in type_data:
                    for d in data_cat[k]:
                        if type(type_data[k][d]) == tuple:
                            content[row][j] = '%s ± %s' % type_data[k][d]
                        else:
                            content[row][j] = type_data[k][d]
                        row = row + 1
        else:
            for i in range(2, len(content)):
                content[i][j] = '-'

    table_style_params = [
            ('SPAN', (0, 0), (-1, 0)),
            ('BACKGROUND', (0, 1), (-1, 1), colors.black),
            ('LINEBELOW', (0, 0), (-1, -1), 0.25, colors.black),
            ('LINEBEFORE', (0, 1), (0, -1), 0.25, colors.black),
            ('LINEAFTER', (-1, 1), (-1, -1), 0.25, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('SIZE', (0, 1), (-1, -1), text_size),
            ('BOTTOMPADDING', (0, 0), (0, 0), 5)
        ]
    start = 2
    cur_label = content[start][0]
    for i in range( start, len(content) ):
        if content[i][0] != cur_label:
            table_style_params.append(
                ('SPAN', (0, start), (0, i - 1))
            )
            cur_label = content[i][0]
            start = i
        elif (i == len(content) - 1 and content[i][0] == cur_label):
            table_style_params.append(
                ('SPAN', (0, start), (0, i))
            )
    # table = Table(content, hAlign=align, colWidths=(A4[0] - page_margin * 2)/(len( content[len(content)-1] ) + 0.5), spaceAfter=spaceAfter)
    table = Table(content, hAlign=align, colWidths=(A4[0]- 0.5 * page_margin) / (len(content[len(content) - 1]) + 0.75), spaceAfter=spaceAfter)
    table.setStyle(table_style_params)

    return table