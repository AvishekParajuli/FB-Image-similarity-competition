import keras
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Input, Lambda
from keras.utils import np_utils
from keras import backend as K
from settings import *  # importing all the variables and Cosntants
from models.loss_funcs import *




def my_norm(ip):
    return K.l2_normalize(ip, axis=-1)


def embedding_model():
    # Simple convolutional model
    # used for the embedding model.
    base_cnn = keras.applications.ResNet50(weights="imagenet", input_shape=IM_SIZE + (3,), include_top=False)
    flatten = keras.layers.Flatten()(base_cnn.output)
    drop1 = keras.layers.Dropout(rate=0.25)(flatten)
    dense1 = keras.layers.Dense(256, activation="relu")(drop1)
    dense1 = keras.layers.BatchNormalization()(dense1)
    output = keras.layers.Dense(256)(dense1)
    output = Lambda(my_norm)(output)

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    mdl = Model(inputs=base_cnn.input, outputs=output, name="Embedding")
    return mdl


def complete_model(base_model, alpha=0.2):
    # Create the complete model with three
    # embedding models and minimize the loss
    # between their output embeddings
    input_1 = Input((imsize, imsize, 3))
    input_2 = Input((imsize, imsize, 3))
    input_3 = Input((imsize, imsize, 3))

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    # A= Lambda(my_norm)(A)
    # P = Lambda(my_norm)(P)
    # N = Lambda(my_norm)(N)

    loss = Lambda(triplet_loss)([A, P, N])
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model


def get_model_name():
    return "resnet50Reg0.8"


def preprocess(x):
    return keras.applications.resnet50.preprocess_input(x)