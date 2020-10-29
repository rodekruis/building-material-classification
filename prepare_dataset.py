"""
Prepare dataset for classifier, splitting images in tran/validation/test sets and per class label
"""

import os, shutil
import pandas as pd
import numpy as np
import click
from PIL import Image
from pathlib import Path

def make_dirs(split, labels):
    for label in labels:
        train_concrete_dir = os.path.join(split, label)
        os.makedirs(train_concrete_dir, exist_ok=True)


@click.command()
@click.option('--input', help='input')
@click.option('--labels', help='labels')
@click.option('--output', help='output')
def main(input, labels, output):

    if not os.path.exists(output):
        os.mkdir(output)

    # Create directories
    train_dir = os.path.join(output, 'train')
    os.makedirs(train_dir, exist_ok=True)
    validation_dir = os.path.join(output, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    test_dir = os.path.join(output, 'test')
    os.makedirs(test_dir, exist_ok=True)

    classes = ['concrete', 'bricks', 'metal', 'thatch']
    make_dirs(train_dir, classes)
    make_dirs(validation_dir, classes)
    make_dirs(test_dir, classes)

    # load image ids
    df = pd.read_csv(labels, index_col=0)
    for class_ in classes:
        ids_class = np.array(df[df['Answer.category.label'] == class_]['Input.image_url'].tolist())
        np.random.shuffle(ids_class)

        # split dataset: 80% train, 10% validation, 10% test
        ids_class_split = np.array_split(ids_class, 10)

        fnames = list(np.concatenate(ids_class_split[:8]))
        for fname in fnames:
            src = Path(input) / fname
            dst = os.path.join(train_dir, class_, os.path.basename(fname))
            shutil.copyfile(src, dst)

        fnames = list(ids_class_split[8])
        for fname in fnames:
            src = Path(input) / fname
            dst = os.path.join(validation_dir, class_, os.path.basename(fname))
            shutil.copyfile(src, dst)

        fnames = list(ids_class_split[9])
        for fname in fnames:
            src = Path(input) / fname
            dst = os.path.join(test_dir, class_, os.path.basename(fname))
            shutil.copyfile(src, dst)

        # Sanity check
        print(f'total training {class_} images: {len(os.listdir(os.path.join(train_dir, class_)))}')
        print(f'total validation {class_} images: {len(os.listdir(os.path.join(validation_dir, class_)))}')
        print(f'total test {class_} images: {len(os.listdir(os.path.join(test_dir, class_)))}')


if __name__ == '__main__':
    main()
