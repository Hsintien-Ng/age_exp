import os
import json
import sys


def write_files(outputFolder, split, data, fileName):
    outputFolder = os.path.join(outputFolder, split)
    if os.path.exists(outputFolder):
        os.makedirs(os.path.join(outputFolder, split))

    filePath = os.path.join(outputFolder, fileName)
    with open(filePath, 'w'):
        json.dump(data, fileName)


def progress_bar(current, total):
    max_arrow = 50
    num_arrow = int(current * max_arrow / total)
    num_line = max_arrow - num_arrow
    bar = '[' + '>' * num_arrow + '-' * num_line + '] ' \
        + '%d/%d' % (current, total)
    if current < total:
        bar += '\r'
    else:
        bar += '\n'
    sys.stdout.write(bar)
    sys.stdout.flush()



if __name__ == '__main__':
    inputFolder = os.path.join('/', 'home', 'xintian', 'projects', 'age_exp', 'MORPH_Split')
    inputFolders = os.path.join('/', 'mnt', 'disk50', 'datasets', 'MORPH', 'Patches_split')
    outputFolder = os.path.join('/', 'home', 'xintian', 'projects', 'age_exp', 'MORPH_Patches')

    split = ['train', 'valid', 'test']
    mode = ['face', 'left_eye', 'right_eye', 'left_cheek', 'right_cheek', 'mouth']

    for s in split:
        if not os.path.exists(os.path.join(outputFolder, s)):
            os.makedirs(os.path.join(outputFolder, s))
        inputPath = os.path.join(inputFolder, s)
        fileNames = os.listdir(inputPath)
        for fN in fileNames:
            filePath = os.path.join(inputPath, fN)
            f = open(filePath, 'r')
            dataPaths = f.read().splitlines()
            f.close()
            for dP in dataPaths:
                imgName = os.path.split(dP)[1]
                imgName, suffix = os.path.splitext(imgName)
                data_dict = {}
                for m in mode:
                    modePath = os.path.join(inputFolders, m)
                    dataPath = os.path.join(modePath, imgName + '.png')
                    data_dict[m] = dataPath
                if not os.path.exists(os.path.join(outputFolder, s, imgName + '.json')):
                    os.mknod(os.path.join(outputFolder, s, imgName + '.json'))
                f = open(os.path.join(outputFolder, s, imgName + '.json'), 'w')
                json.dump(data_dict, f)
                f.close()
        print('%s Done!' % s)