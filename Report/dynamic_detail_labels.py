import content as mcontent
import colors as mcolor
from reportlab.graphics.shapes import Drawing, Rect, colors

SCALE_FACTOR = 3
RIGHT_MARGIN = 30

def content(data):
    content = []

    for t in data:
        d = Drawing(SCALE_FACTOR * 100 + RIGHT_MARGIN, 10)
        # d.add(Rect(0, 0, SCALE_FACTOR * 10, 10, fillColor=colors.lightgrey, strokeColor=colors.transparent))
        value = t
        if t == 'Horizontal':
            d.add(Rect(0, 0, SCALE_FACTOR, 10, fillColor=mcolor.DATA_RED, strokeColor=colors.transparent))
        elif t == 'Vertical':
            d.add(Rect(0, 0, SCALE_FACTOR, 10, fillColor=mcolor.DATA_BLUE, strokeColor=colors.transparent))
        elif t == 'Spot_Horizontal':
            d.add(Rect(0, 0, SCALE_FACTOR, 10, fillColor=mcolor.DATA_LIGHT_RED, strokeColor=colors.transparent))
        elif t == 'Spot_Vertical':
            d.add(Rect(0, 0, SCALE_FACTOR, 10, fillColor=mcolor.DATA_LIGHT_BLUE, strokeColor=colors.transparent))
        content.append( value )

    # content[0][0] = mcontent.DYNAMIC_DETAIL_LABELS
    return content