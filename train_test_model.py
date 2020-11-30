"""
train and test classifier
"""

import os
import click
import pandas as pd
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D
from PIL import ImageFile
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
ImageFile.LOAD_TRUNCATED_IMAGES = True

HEIGHT = 256
WIDTH = 256


@click.command()
@click.option('--input_images_dir', default='data', help='input')
@click.option('--batch_size', default=8, help='number of images per batch')
@click.option('--num_epochs', default=15, help='number of training epochs')
@click.option('--learning_rate', default=1e-4, help='learning rate of the optimizer')
@click.option('--save_plot_training', default=False, help='save plots of accuracy and loss vs training epochs')
@click.option('--inference', default=False, help='do inference')
def main(input_images_dir, batch_size, num_epochs, learning_rate, save_plot_training, inference):

    RUN_DIR = f'runs/run_ep{15}_bs{batch_size}_lr{learning_rate}_'+ datetime.now().strftime("%m%d%Y_%H%M%S")
    os.makedirs(RUN_DIR, exist_ok=True)

    # 1. PREPARE INPUT DATA

    # set directory for train and validation data
    TRAIN_DIR = input_images_dir+"/train"
    VALID_DIR = input_images_dir+"/validation"
    TEST_DIR = input_images_dir + "/test"
    class_list = ["bricks", "concrete", "metal", "thatch"]

    # generate batches of training data (w/ augmentation)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        horizontal_flip=True
        )
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size=(HEIGHT, WIDTH),
                                                        batch_size=batch_size,
                                                        interpolation='hamming',
                                                        shuffle=False)

    # generate batches of validation data (w/o augmentation)
    validation_datagen = ImageDataGenerator(
          preprocessing_function=preprocess_input
        )
    validation_generator = validation_datagen.flow_from_directory(VALID_DIR,
                                                        target_size=(HEIGHT, WIDTH),
                                                        batch_size=batch_size,
                                                        interpolation='hamming',
                                                        shuffle=False)

    # generate batches of test data (w/o augmentation)
    test_datagen = ImageDataGenerator(
          preprocessing_function=preprocess_input
        )
    test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                      target_size=(HEIGHT, WIDTH),
                                                      batch_size=batch_size,
                                                      interpolation='hamming',
                                                      shuffle=False)

    # 2. BUILD MODEL

    # load ResNet50 pre-trained model
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=(HEIGHT, WIDTH, 3))
    # freeze bottom layer (not re-trainable)
    for layer in base_model.layers:
        layer.trainable = False

    # model hyperparameters
    nodes_per_layer = [64]
    dropout = 0.2

    # build model
    finetune_model = build_finetune_model(base_model,
                                          dropout=dropout,
                                          nodes_per_layer=nodes_per_layer,
                                          num_classes=len(class_list))
    print(finetune_model.summary())

    # 3. TRAIN MODEL

    optimizer = Adam(lr=learning_rate)
    finetune_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    # callbacks_list = [checkpoint]

    history = finetune_model.fit(train_generator,
                                   epochs=num_epochs,
                                   workers=6,
                                   steps_per_epoch=train_generator.samples // batch_size,
                                   validation_data=validation_generator,
                                   validation_steps=validation_generator.samples // batch_size,
                                   shuffle=False,
                                   # callbacks=callbacks_list,
                                   use_multiprocessing=False)
    # plot training history
    if save_plot_training:
        plot_training(history, f'{RUN_DIR}/plots')

    # 4. TEST MODEL
    Y_pred = finetune_model.predict(test_generator,
                                    steps=test_generator.samples // batch_size + 1,
                                    workers=6,
                                    use_multiprocessing=False)
    df = pd.DataFrame(data=Y_pred, columns=class_list)
    df.to_csv(f'{RUN_DIR}/test_predict.csv')

    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=class_list))

    # 5. SAVE MODEL
    finetune_model.save(f'{RUN_DIR}/model')

    # 6. INFERENCE ON ALL DATA
    if inference:
        INFERENCE_DIR = input_images_dir + "/inference"
        inference_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        inference_generator = test_datagen.flow_from_directory(INFERENCE_DIR,
                                                               target_size=(HEIGHT, WIDTH),
                                                               batch_size=batch_size,
                                                               interpolation='hamming',
                                                               shuffle=False)
        Y_pred = finetune_model.predict(inference_generator,
                                        steps=inference_generator.samples // batch_size + 1,
                                        workers=6,
                                        use_multiprocessing=False)
        # save results
        # Y_pred.save(f'{RUN_DIR}/inference/')


def build_finetune_model(base_model, dropout, nodes_per_layer, num_classes):
    """ add dense layers + 1 softmax layer to a base model """

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    for nodes_in_layer in nodes_per_layer:
        x = Dense(nodes_in_layer, activation='relu')(x)
        if dropout > 0.:
            x = Dropout(rate=dropout)(x)

    # Final softmax layer for classification
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model


def plot_training(history, savedir):
    """ Plot the train / validation accuracy and loss """

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig(f'{savedir}/acc_vs_epochs.png')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig(f'{savedir}/loss_vs_epochs.png')


if __name__ == '__main__':
    main()
