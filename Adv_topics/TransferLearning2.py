import os, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array


AUTOTUNE = tf.data.AUTOTUNE


def setDataset(path):

    PATH = 'training_dataset/chest_xray'
    train_dir = os.path.join(path, 'train')
    validation_dir = os.path.join(path, 'val')
    test_dir = os.path.join(path, 'test')

    BATCH_SIZE = 32
    IMG_SIZE = (224,224)

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                    shuffle=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)

    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    class_names = train_dataset.class_names

    print (class_names)

    print('Number of training batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    return train_dataset, validation_dataset, test_dataset

def makeModel():
    # Load the ResNet50 Conv Layers only along with the pre-training filter
    base_model = ResNet50V2(include_top=False, weights='imagenet')

    # Fix the weights for the conv layers - this won't get updated during training.  
    base_model.trainable = False

    # Setup some basic data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    # grab the image processor function for resnet_v2
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    inputs = tf.keras.layers.Input(shape=(224,224, 3))

    # Build the moddel 
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.models.Sequential([inputs, outputs])

    # resnet = ResNet50V2(include_top=False, weights='imagenet')
    # for layer in resnet.layers:
    #     layer.trainable = False

    # fc1 = tf.keras.layers.Dense(100)(resnet.layers[-1].output)
    # fc2 = tf.keras.layers.Dense(100)(fc1)
    # logits = tf.keras.layers.Dense(2)(fc2)
    # output = tf.keras.layers.Activation('softmax')(logits)
    # model = tf.keras.Model(resnet.input, output)

    # model.summary()

    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.RandomFlip('horizontal'),
    #     tf.keras.layers.RandomRotation(0.2),
    # ])

    # # grab the image processor function for resnet_v2
    # inputs = tf.keras.Input(shape=(224,224, 3))
    # preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    # x = data_augmentation(inputs)
    # x = preprocess_input(x)
    # model = tf.keras.Model(inputs, model.layers[-1].output)

    model.summary()

    return model

def transferLearn(train_dataset, validation_dataset, epoch=10):

    # Use disk pre-fetching to avoid IO bloking during training
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    model = makeModel()

    # Compile Model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    model.summary()

    initial_epochs = epoch
    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    checkpoint_path = "cp-{epoch:04d}.weights.h5"
    # Calculate the number of batches per epoch
    import math
    n_batches = len(train_dataset) / 32 #BATCH_SIZE
    n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=10*n_batches)

    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        callbacks=[cp_callback],
                        validation_data=validation_dataset)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return model


def loadModel(weights):

    model = makeModel()

    model.load_weights(weights)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    return model



#####################################################################
# Main 
#####################################################################
if __name__ == "__main__":

    PATH = 'training_dataset/chest_xray'
    trainDs, ValDs, testDs = setDataset(PATH)

    model = transferLearn(trainDs, ValDs, epoch=12)
    #model = loadModel("cp-0012.weights.h5")

    testDs = testDs.prefetch(buffer_size=AUTOTUNE)

    loss, accuracy = model.evaluate(testDs)
    print('Test accuracy :', accuracy)
