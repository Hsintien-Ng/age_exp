import os
import cv2

src_dir = os.path.join('/', 'home', 'yjfu', 'AffectNet', 'Manually_Annotated_Images')
target_dir = os.path.join('/', 'home', 'yjfu', 'AffectNet', 'Resized_Images')


def padding_and_resize(path, size):
    img = None
    try:
        img = cv2.imread(path)
        shape = img.shape
        longger = max(shape)
        img = cv2.copyMakeBorder(img, (longger-shape[0])/2, (longger-shape[0])/2,
                                 (longger - shape[1]) / 2, (longger-shape[1])/2,
                                 cv2.BORDER_REPLICATE)
        img = cv2.resize(img, size)
    except:
        print('error file')
    return img

def copy_and_resize(src, target):
    all_files = os.listdir(src)
    for file in all_files:
        src_abs_path = os.path.join(src, file)
        target_abs_path = os.path.join(target, file)
        if os.path.isdir(src_abs_path):
            if not os.path.exists(target_abs_path):
                os.makedirs(target_abs_path)
            copy_and_resize(src_abs_path, target_abs_path)
        else:
            img = padding_and_resize(src_abs_path, (300, 300))
            if img is not None:
                cv2.imwrite(target_abs_path, img)


copy_and_resize(src_dir, target_dir)