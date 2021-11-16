import content as mcontent
import colors as mcolor
from reportlab.graphics.shapes import Drawing, Rect, colors

SCALE_FACTOR = 3
RIGHT_MARGIN = 30

def content(data):
    content = []
    
    for t in mcontent.RISK_RESULT_TYPE:
        d = Drawing(SCALE_FACTOR * 100 + RIGHT_MARGIN, 10)
        d.add(Rect(0, 0, SCALE_FACTOR * 100, 10, fillColor=colors.lightgrey, strokeColor=colors.transparent))
        value = data[t]
        if t == 'Normal':
            d.add(Rect(0, 0, value * SCALE_FACTOR, 10, fillColor=mcolor.DATA_GREEN, strokeColor=colors.transparent))
        elif t == 'Benign disorder':
            d.add(Rect(0, 0, value * SCALE_FACTOR, 10, fillColor=mcolor.DATA_YELLOW, strokeColor=colors.transparent))
        elif t == 'Dangerous disorder':
            d.add(Rect(0, 0, value * SCALE_FACTOR, 10, fillColor=mcolor.DATA_RED, strokeColor=colors.transparent))
        content.append( [ '', '%s:' % (t), '%.2f %%' % (value), d ] )

    content[0][0] = mcontent.RISK_RESULT
    return content
