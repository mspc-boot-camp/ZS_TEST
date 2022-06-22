

def read_imagepath_from_txt(image_path):
    f=open(image_path, encoding='gbk')
    txt=[]
    for line in f:
        txt.append(line.strip())
    print(txt)
    return txt


def find_label(str):
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '-' and str[i - 1] == 'g':
            name = 'dog'
            break
        if str[i] == '-' and str[i - 1] == 't':
            name = 'cat'
            break

    if name == 'dog':
        return 1
    else:
        return 0

