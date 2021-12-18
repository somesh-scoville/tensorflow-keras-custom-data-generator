"""TensorFlow Keras custom data generator"""
import os
import random
from collections import Counter

import cv2
import h5py
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels,  batch_size, file_paths=None, augment=None, shuffle=False):
        self.indexes = np.arange(len(images))
        self.images = images
        self.labels = labels
        self.augment = augment
        self.shuffle = shuffle
        self.all_file_paths = file_paths
        self.file_paths = None
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        self.file_paths = [self.all_file_paths[k] for k in indexes]
        images = [self.images[k] for k in indexes]
        labels = np.array([self.labels[k] for k in indexes])

        if self.augment:
            images = augmentation(images)

        images = np.array([preprocess_image(image) for image in images]).astype(dtype='float32')

        return images, labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.all_images))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def augmentation(images):
    """Apply batch wise data augmentation using imgaug library"""
    seq = iaa.Sequential(
        [iaa.Fliplr(0.2), iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode='constant', pad_cval=0),
         iaa.Cutout(nb_iterations=1, size=0.3, squared=False, fill_mode="constant", cval=0),
         ],
        random_order=False
    )
    return seq.augment_images(images)

def preprocess_image(image):
    """implement preprocessing according to the use case"""
    image = image / 255.0
    return image


def prepare_image_data(label_file_path, size, shuffle=False):
    """returns images, labels and image paths

    :param str label_file_path: path or name of label text file which contains image file names and labels.
    :param tuple size: input image size to model (width, height)
    :param bool shuffle: if true shuffle images within the batch for each iteration

    :return images, labels and image paths
    :rtype: ndarray, ndarray, list
    """
    with open(label_file_path) as file:
        lines = file.readlines()
    if shuffle:
        random.shuffle(lines)
    print("%s images to embed" % len(lines))

    images, labels, image_paths = [], [], []
    for line in tqdm(lines):
        image_path, label = line.strip().split(' ')  # first column is image name second is label, "path/name.jpg 1"
        if not os.path.exists(image_path):
            continue

        # If system runs out of memory for large dataset use image path instead of image array
        image = cv2.imread(image_path)
        if max(image.shape) < max(size):
            image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)  # size = width, height
        else:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)  # size = width, height

        images.append(image)
        labels.append(int(label))
        image_paths.append(image_path)

    images = np.array(images)
    labels = np.array(labels)
    image_paths = image_paths
    return images, labels, image_paths


def get_dataset(dataset_path, size, shuffle=False):
    """get data from already saved h5py file or generate new.

    :param str dataset_path: path or name of label text file which contains image file names and labels
    \or path to h5py file or path to image directory
    :param tuple size: input image size (width, height)
    :param bool shuffle: if true shuffle images within the batch for each iteration

    :return image arrays, labels, image_paths
    :rtype ndarray, ndarray, ndarray
    """
    if not os.path.exists(dataset_path):
        print(f'{dataset_path} path does not exists')
        return None, None, None

    images, labels, file_paths = np.empty(shape=0), np.empty(shape=0), np.empty(shape=0)

    if os.path.isfile(dataset_path) and os.path.basename(dataset_path).split('.')[1] == 'h5':
        hf = h5py.File(dataset_path, 'r')
        labels = np.array(hf.get('labels'))
        images = np.array(hf.get('images')) if hf.get('images') is not None else images
        file_paths = np.array(hf.get('file_paths')) if hf.get('file_paths') is not None else file_paths
        hf.close()
        return images, labels, file_paths

    if os.path.isdir(dataset_path):
        from dataset.prepare_dataset import prepare_dataset_txt
        dataset_txt_path = os.path.join(os.path.dirname(dataset_path), 'data.txt')
        prepare_dataset_txt(dataset_path, dataset_txt_path)
        dataset_path = dataset_txt_path

    images, labels, file_paths = prepare_image_data(dataset_path, size, shuffle)
    dataset_path = dataset_path.replace('.txt', '.h5')

    hf = h5py.File(dataset_path, 'w')
    hf.create_dataset('images', data=images)
    hf.create_dataset('labels', data=labels)
    ascii_image_paths = [n.encode("ascii", "ignore") for n in file_paths]
    hf.create_dataset('file_paths', (len(ascii_image_paths), 1), 'S10', ascii_image_paths)
    hf.close()

    return images, labels, file_paths


def get_data_generator(dataset_path, size, batch_size, shuffle, augment, writer=None):
    """ Prepare train, val, & test data loaders
    Augment training data using:cropping, shifting (vertical/horizontal), horizontal flipping, rotation

    :param str dataset_path: path label text file which contains image file names and labels
    \or path to h5py file or path to image directory
    :param tuple size: input image size (width, height)
    :param int batch_size: batch size
    :param bool shuffle: if true shuffle images within the batch for each iteration
    :param bool augment: augmentation
    :param writer writer: tensorboard event file writer

    :return custom data loader
    :rtype Dataloader
    """
    image_data, labels, file_paths = get_dataset(dataset_path, size, shuffle)
    if image_data is None:
        return

    fig = plt.figure()
    title = os.path.basename(dataset_path).split('.')[0]
    plt.title('%s dataset' % title)
    d = dict(sorted(Counter(labels).items(), key=lambda item: item[0]))
    plt.bar(range(len(d)), list(d.values()), color=np.random.uniform(0, 1, size=(len(d), 3)))
    plt.xticks(range(len(d)), list(d.keys()))
    plt.show()
    if writer is not None:
        writer.add_figure('%s dataset' % title, fig)

    data_generator = DataGenerator(image_data, labels, batch_size, file_paths, augment, shuffle)
    return data_generator


def sanity_check(data_loader):
    plt.figure(figsize=(20, 10))
    print(f'\ndataset size: {len(data_loader.images)} images')
    for i, data in enumerate(data_loader):
        inputs, labels = data

        print(f'input batch shape: {inputs.shape}, \nlabels batch shape:{labels.shape}')
        print(f'input array range: {np.min(inputs)} to {np.max(inputs)}\n')

        img = inputs[:5, :, :, :] * 255.0
        img = np.hstack(img).astype(np.int64)
        if img.shape[-1] == 1:
            plt.imshow(img, cmap='gray'), plt.show()
        else:
            plt.imshow(img[:,:,::-1]), plt.show()
        break


if __name__ == "__main__":
    path = "fruits/images"
    # path = "fruits/data.txt"
    # path = "fruits/data.h5"
    # path = "fruits/train.txt"
    dummy_data_generator = get_data_generator(dataset_path=path, size=(96, 96), batch_size=64, shuffle=True,
                                              augment=True, writer=None)

    if dummy_data_generator is not None:
        sanity_check(dummy_data_generator)
