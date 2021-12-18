"""Prepare dataset text file"""
import glob
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ['prepare_dataset_txt', 'split_classification_dataset', 'get_label_from_filepath']


def get_label_from_filepath(file_path, label_map=None):
    """return label from filepath, change this function according to the filenames in the dataset"""
    label = os.path.basename(file_path).split('.')[0].split('_')[-1]  # 'dir_path/images/image1_mango.jpg'
    if label_map is None:
        label_map = {'mango': 0, 'guava': 1, 'banana': 2, 'cocos': 3, 'papaya': 4, 'grapepink': 5, 'orange': 6,
                     'pineapple': 7, 'pear': 8, 'kiwi': 9}
    label = label_map[label]  # 'dir_path/images/image1_class2.jpg'
    label = int(label)
    return label


def prepare_dataset_txt(image_dir, txt_path=None):
    """generate text file for image paths and labels"""
    file_paths = sorted(glob.glob(image_dir + "/*.jpg") + glob.glob(image_dir + "/*.png"))
    images, labels, image_n_labels = [], [], []
    for file_path in file_paths:
        label = get_label_from_filepath(file_path)
        image_n_labels += [[file_path, label]]
    data = pd.DataFrame(image_n_labels)
    if txt_path is None:
        txt_path = os.path.join(os.path.dirname(image_dir), 'data.txt')
    data.to_csv(txt_path, header=False, index=False, sep=' ')


def split_classification_dataset(image_dir, label_map, trainval_fraction=0.9, train_fraction=0.8, random_seed=2):
    """split dataset into calss wise balanced train val and test sets using splitting ratios and save
    'trainval.txt','train.txt','val.txt','test.txt' files with columns of filename space label.
    """
    all_files = glob.glob(image_dir + '/*.jpg') + glob.glob(image_dir + '/*.png')
    if len(all_files) == 0:
        print('no images found in %s' % image_dir)
        return

    trainval_files, train_files, test_files, val_files = [], [], [], []

    for label in label_map.keys():
        classwise_files = [file for file in all_files if label_map[label] == get_label_from_filepath(file, label_map)]
        classwise_files = [[i, label_map[label]] for i in classwise_files]
        tv_files, te_files = train_test_split(classwise_files, test_size=1-trainval_fraction, random_state=random_seed)
        tr_files, v_files = train_test_split(tv_files, test_size=trainval_fraction-train_fraction,
                                             random_state=random_seed)
        trainval_files += tv_files
        train_files += tr_files
        val_files += v_files
        test_files += te_files

    random.shuffle(trainval_files), random.shuffle(train_files), random.shuffle(val_files), random.shuffle(test_files)
    all_files = [trainval_files, train_files, val_files, test_files]

    colors = np.random.uniform(0, 1, size=(len(label_map), 3))
    names = ['trainval.txt', 'train.txt', 'val.txt', 'test.txt']

    for i in range(4):
        path = os.path.join(os.path.dirname(image_dir), names[i])
        data = pd.DataFrame(all_files[i])
        data.to_csv(path, header=False, index=False, sep=' ')
        data.columns = ['filename', 'label']
        # print("%s images split with %s" % (names[i].split('.')[0], dict(data['label'].value_counts())))
    print(f"Data split completed and files saved at {os.path.dirname(image_dir)}")


if __name__ == "__main__":
    imagedir = "fruits/images"
    textpath = "fruits/data.txt"
    prepare_dataset_txt(imagedir, textpath)
    labelmap = {'mango': 0, 'guava': 1, 'banana': 2, 'cocos': 3, 'papaya': 4, 'grapepink': 5, 'orange': 6,
                 'pineapple': 7, 'pear': 8, 'kiwi': 9}
    split_classification_dataset(imagedir, labelmap, trainval_fraction=0.8, train_fraction=0.6, random_seed=2)
