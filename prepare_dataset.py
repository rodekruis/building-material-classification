"""
Prepare dataset for classifier, splitting images in tran/validation/test sets and per class label
"""

import os, shutil
import pandas as pd
import numpy as np
import plac

@plac.annotations(
    input_images_dir="input data directory: all images, unlabeled",
    labels_file="csv file with class labels",
    output_images_dir="output data directory: images separated by train/validation/test and class label",
)
def main(input_images_dir, labels_file, output_images_dir):

    if not os.path.exists(output_images_dir):
        os.mkdir(output_images_dir)

    # Create directories
    train_dir = os.path.join(output_images_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    validation_dir = os.path.join(output_images_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    test_dir = os.path.join(output_images_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_concrete_dir = os.path.join(train_dir, 'concrete')
    if not os.path.exists(train_concrete_dir):
        os.mkdir(train_concrete_dir)

    train_bricks_dir = os.path.join(train_dir, 'bricks')
    if not os.path.exists(train_bricks_dir):
        os.mkdir(train_bricks_dir)

    validation_concrete_dir = os.path.join(validation_dir, 'concrete')
    if not os.path.exists(validation_concrete_dir):
        os.mkdir(validation_concrete_dir)

    validation_bricks_dir = os.path.join(validation_dir, 'bricks')
    if not os.path.exists(validation_bricks_dir):
        os.mkdir(validation_bricks_dir)

    test_concrete_dir = os.path.join(test_dir, 'concrete')
    if not os.path.exists(test_concrete_dir):
        os.mkdir(test_concrete_dir)

    test_bricks_dir = os.path.join(test_dir, 'bricks')
    if not os.path.exists(test_bricks_dir):
        os.mkdir(test_bricks_dir)

    # load image ids
    df = pd.read_csv(labels_file, index_col=0)
    ids_concrete = np.array(df[df['Answer.category.label'] == 'concrete']['Input.image_url'].tolist())
    ids_bricks = np.array(df[df['Answer.category.label'] == 'bricks']['Input.image_url'].tolist())
    np.random.shuffle(ids_concrete)
    np.random.shuffle(ids_bricks)

    # split dataset: 80% train, 10% validation, 10% test
    ids_concrete_split = np.array_split(ids_concrete, 10)
    ids_bricks_split = np.array_split(ids_bricks, 10)

    fnames = list(np.concatenate(ids_concrete_split[:8]))
    for fname in fnames:
        src = os.path.join(input_images_dir, fname)
        dst = os.path.join(train_concrete_dir, fname)
        shutil.copyfile(src, dst)

    fnames = list(ids_concrete_split[8])
    for fname in fnames:
        src = os.path.join(input_images_dir, fname)
        dst = os.path.join(validation_concrete_dir, fname)
        shutil.copyfile(src, dst)

    fnames = list(ids_concrete_split[9])
    for fname in fnames:
        src = os.path.join(input_images_dir, fname)
        dst = os.path.join(test_concrete_dir, fname)
        shutil.copyfile(src, dst)

    fnames = list(np.concatenate(ids_bricks_split[:8]))
    for fname in fnames:
        src = os.path.join(input_images_dir, fname)
        dst = os.path.join(train_bricks_dir, fname)
        shutil.copyfile(src, dst)

    fnames = list(ids_bricks_split[8])
    for fname in fnames:
        src = os.path.join(input_images_dir, fname)
        dst = os.path.join(validation_bricks_dir, fname)
        shutil.copyfile(src, dst)

    fnames = list(ids_bricks_split[9])
    for fname in fnames:
        src = os.path.join(input_images_dir, fname)
        dst = os.path.join(test_bricks_dir, fname)
        shutil.copyfile(src, dst)

    # Sanity checks
    print('total training concrete images:', len(os.listdir(train_concrete_dir)))
    print('total training bricks images:', len(os.listdir(train_bricks_dir)))
    print('total validation concrete images:', len(os.listdir(validation_concrete_dir)))
    print('total validation bricks images:', len(os.listdir(validation_bricks_dir)))
    print('total test concrete images:', len(os.listdir(test_concrete_dir)))
    print('total test bricks images:', len(os.listdir(test_bricks_dir)))


if __name__ == '__main__':
    plac.call(main)
