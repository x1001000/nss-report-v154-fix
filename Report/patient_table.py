import content as mcontent

def content(data):
    content = []

    patient_content = []
    patient_content.append( '%s: %s' % (mcontent.PATIENT_DATA[0], data[mcontent.PATIENT_DATA[0]]))
    patient_content.append( '%s: %s' % (mcontent.PATIENT_DATA[1], data[mcontent.PATIENT_DATA[1]]))
    patient_content.append( '%s: %s' % (mcontent.PATIENT_DATA[2], data[mcontent.PATIENT_DATA[2]]))
    
    location_content = []
    location_content.append('')
    location_content.append( '%s: %s' % (mcontent.PATIENT_DATA[3], data[mcontent.PATIENT_DATA[3]]))
    location_content.append( '%s: %s' % (mcontent.PATIENT_DATA[4], data[mcontent.PATIENT_DATA[4]]))

    device_content = []
    device_content.append( '%s: %s' % (mcontent.PATIENT_DATA[5], data[mcontent.PATIENT_DATA[5]]))
    
    exam_content = []
    exam_content.append( '%s: %d VOR tests includes: %s' % (mcontent.PATIENT_DATA[6], len(mcontent.TEST_TYPE), str(data[mcontent.PATIENT_DATA[6]])))

    content.append(patient_content)
    content.append(location_content)
    content.append(device_content)
    content.append(exam_content)

    return content