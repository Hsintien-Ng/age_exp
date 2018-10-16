import os
from PIL import Image


def crop_margin(img_path, margin):
    """
    crop the image with given margin
    :param img_path:
    :param margin: the coordinate of top-left and bottom-right of region
            of interest
    :return: cropped image, an instance of Image
    """
    img = Image.open(img_path, 'r')
    img = img.crop(margin)
    return img


def generate_cropped_subject(origin_subject_dir, cropped_subject_dir):
    """
    crop all image in origin_subject_dir, then save them into cropped_subject_dir.
    Note that the subject_dir must be managed properly, i.e. it should contain
    several video directory containing images.
    :param origin_subject_dir:
    :param cropped_subject_dir:
    :return:
    """
    if not os.path.exists(cropped_subject_dir):
        os.mkdir(cropped_subject_dir)
    video_list = os.listdir(origin_subject_dir)
    for video in video_list:
        if '.' not in video:
            origin_video_dir = os.path.join(origin_subject_dir, video)
            cropped_video_dir = os.path.join(cropped_subject_dir, video)
            generate_cropped_video(origin_video_dir, cropped_video_dir)


def generate_cropped_video(origin_video_dir, cropped_video_dir):
    """
    crop all image in origin_video_dir, then save them into cropped_video_dir
    Note that there should only be images(.png) in cropped_video_dir.
    :param origin_video_dir:
    :param cropped_video_dir:
    :return:
    """
    if not os.path.exists(cropped_video_dir):
        os.mkdir(cropped_video_dir)
    img_list = os.listdir(origin_video_dir)
    for img in img_list:
        if 'png' not in img:
            continue
        origin_img_path = os.path.join(origin_video_dir, img)
        cropped_img_path = os.path.join(cropped_video_dir, img)

        cropped_img = crop_margin(origin_img_path, margin=[0, 10, 640, 435])
        cropped_img.save(cropped_img_path)


def generate_cropped_dataset(origin_dir, target_dir):
    """
    crop all image in origin_dir, then save them into cropped_dir.
    Note that the subject_dir must be managed properly, i.e. it should contain
    several subject directory containing videos.
    :param origin_dir:
    :param target_dir:
    :return:
    """
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    subjects_list = os.listdir(origin_dir)
    for subject in subjects_list:
        origin_subject_dir = os.path.join(origin_dir, subject)
        cropped_subject_dir = os.path.join(target_dir, subject)
        generate_cropped_subject(origin_subject_dir, cropped_subject_dir)


if __name__ == '__main__':
    work_dir = os.path.join('/', 'home', 'yjfu', 'expression_datasets')
    CKP_dir = os.path.join(work_dir, 'CK+')

    origin_dir = os.path.join(CKP_dir, 'cohn-kanade-images')
    target_dir = os.path.join(CKP_dir, 'cropped_images')

    generate_cropped_dataset(origin_dir, target_dir)
