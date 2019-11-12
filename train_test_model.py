"""
train and test classifier
"""

import plac
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
ImageFile.LOAD_TRUNCATED_IMAGES = True

HEIGHT = 224
WIDTH = 224


@plac.annotations(
    input_images_dir="input data directory: images separated by train/validation/test and label",
    batch_size="number of images per batch",
    num_epochs="number of training epochs",
    learning_rate="learning rate of the optimizer",
    save_plot_training="save plots of accuracy and loss vs training epochs"
)
def main(input_images_dir='data', batch_size=8, num_epochs=100, learning_rate=1e-4, save_plot_training=False):

    # 1. PREPARE INPUT DATA

    # set directory for train and validation data
    TRAIN_DIR = input_images_dir+"/train"
    VALID_DIR = input_images_dir+"/validation"
    TEST_DIR = input_images_dir + "/test"
    class_list = ["bricks", "concrete"]

    # generate batches of training data (w/ augmentation)
    train_datagen = ImageDataGenerator(
          preprocessing_function=preprocess_input,
          rotation_range=30,
          horizontal_flip=True
        )
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size=(HEIGHT, WIDTH),
                                                        batch_size=batch_size)

    # generate batches of validation data (w/o augmentation)
    validation_datagen = ImageDataGenerator(
          preprocessing_function=preprocess_input
        )
    validation_generator = validation_datagen.flow_from_directory(VALID_DIR,
                                                        target_size=(HEIGHT, WIDTH),
                                                        batch_size=batch_size)

    # generate batches of test data (w/o augmentation)
    test_datagen = ImageDataGenerator(
          preprocessing_function=preprocess_input
        )
    test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                      target_size=(HEIGHT, WIDTH),
                                                      batch_size=batch_size)

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
    dropout = 0.

    # build model
    finetune_model = build_finetune_model(base_model,
                                          dropout=dropout,
                                          nodes_per_layer=nodes_per_layer,
                                          num_classes=len(class_list))

    # 3. TRAIN MODEL

    optimizer = Adam(lr=learning_rate)
    finetune_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
    # checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    # callbacks_list = [checkpoint]

    history = finetune_model.fit_generator(train_generator,
                                           epochs=num_epochs,
                                           workers=1,
                                           steps_per_epoch=train_generator.samples // batch_size,
                                           validation_data=validation_generator,
                                           validation_steps=validation_generator.samples // batch_size,
                                           shuffle=False,
                                           # callbacks=callbacks_list,
                                           use_multiprocessing=False)
    # plot training history
    if save_plot_training:
        plot_training(history)

    # 4. TEST MODEL

    Y_pred = finetune_model.predict_generator(test_generator,
                                              steps=test_generator.samples // batch_size + 1,
                                              workers=1,
                                              use_multiprocessing=False)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=class_list))


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


def plot_training(history):
    """ Plot the train / validation accuracy and loss """

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig('acc_vs_epochs.png')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('loss_vs_epochs.png')


if __name__ == '__main__':
    plac.call(main)
