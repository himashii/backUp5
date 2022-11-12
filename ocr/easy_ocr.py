import easyocr


def get_ocr(path, lang):
    reader = easyocr.Reader([lang])
    result = reader.readtext('upload/' + path)
    for i in result:
        print(i[1])
    if result[0][1] is not None:
        return result[0][1]
    else:
        return 'Did not found any text by ocr'
