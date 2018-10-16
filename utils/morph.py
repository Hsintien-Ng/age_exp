import os
import random


def write_files(output_path, images_path, mode, age_all):
    dataFolder = os.path.join(output_path, mode)
    if not os.path.exists(dataFolder):
        os.mkdir(dataFolder)

    for i in range(len(age_all)):
        age_list = age_all[i]
        print('age_{}_{}: '.format(i*10, i*10+9), len(age_list))
        if len(age_list) > 0:
            random.shuffle(age_list)

            fileName = os.path.join(dataFolder, 'age_{}_{}.txt'.format(i*10, i*10+9))
            if not os.path.exists(fileName):
                os.mknod(fileName)

            # write down in txt file.
            fp = open(fileName, 'w+')
            for imgName in age_list:
                imgPath = os.path.join(images_path, imgName)
                fp.write(imgPath + '\n')
            fp.close()


if __name__ == '__main__':
    dataset_path = '/mnt/disk50/datasets/MORPH'
    images_path = os.path.join(dataset_path, 'Images_ori')
    output_path = os.path.join('/', 'home', 'xintian', 'projects', 'age_exp', 'MORPH_Small_Scale')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    age_0_9 = []
    age_10_19 = []
    age_20_29 = []
    age_30_39 = []
    age_40_49 = []
    age_50_59 = []
    age_60_69 = []
    age_70_79 = []
    age_80_89 = []
    age_90_99 = []
    age_all = [age_0_9, age_10_19, age_20_29, age_30_39, age_40_49,
               age_50_59, age_60_69, age_70_79, age_80_89, age_90_99]

    for file in os.listdir(images_path):
        if file[-4:] == '.JPG':
            age = int(file[-6: -4])
            if len(age_all[int(age / 10)]) <= 1000:
                age_all[int(age / 10)].append(file)

    # print statistic results.
    train_all = []
    valid_all = []
    test_all = []
    for i in range(len(age_all)):
        age_list = age_all[i]
        print('age_{}_{}: '.format(i*10, i*10+9), len(age_list))
        if len(age_list) > 0:
            random.shuffle(age_list)
            train_sel = int(len(age_list) * 0.8)
            val_sel = int(len(age_list) * 0.9)
            train_all.append(age_list[:train_sel])
            valid_all.append(age_list[train_sel:val_sel])
            test_all.append(age_list[val_sel:])
        else:
            train_all.append([])
            valid_all.append([])
            test_all.append([])

    for mode in ['train', 'valid', 'test']:
        if mode == 'train':
            age_all = train_all
        elif mode == 'valid':
            age_all = valid_all
        else:
            age_all = test_all
        write_files(output_path, images_path, mode, age_all)