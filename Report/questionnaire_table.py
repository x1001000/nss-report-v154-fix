import content as mcontent

def content(data):
    content = []

    title_content = [ mcontent.QUESTIONNAIRE ]
    title_content.extend(mcontent.QUESTIONNAIRE_TYPE)

    data_content = [ '' ]
    for t in mcontent.QUESTIONNAIRE_TYPE:
        data_content.append( data[t] )

    content.append(title_content)
    content.append(data_content)

    return content