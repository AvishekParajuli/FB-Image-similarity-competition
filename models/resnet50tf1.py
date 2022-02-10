import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Input, Lambda
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import backend as K
from settings import *  # importing all the variables and Cosntants
from models.loss_funcs import *




def my_norm(ip):
    return K.l2_normalize(ip, axis=-1)

def embedding_model_old(drop=0.3):
    # Simple convolutional model
    # used for the embedding model.
    base_cnn = tf.keras.applications.ResNet50(weights="imagenet", input_shape=IM_SIZE + (3,), include_top=False)
    flatten = Flatten()(base_cnn.output)
    drop1 = Dropout(rate=drop)(flatten)
    dense1 = Dense(256, activation="relu")(drop1)
    dense1 = BatchNormalization()(dense1)
    output = Dense(256)(dense1)
    output = Lambda(my_norm)(output)

    trainable = False
    count=0
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            print("conv5_block found")
            layer.trainable = True
        layer.trainable = trainable
        if count >143:
            layer.trainable = True
        count+= 1
    print("total no of layers: ", count)

    mdl = Model(inputs=base_cnn.input, outputs=output, name="Embedding")
    return mdl

def embedding_model_new(drop=0.3,finetune=False):
    # Simple convolutional model
    # used for the embedding model.
    base_cnn = tf.keras.applications.ResNet50(weights="imagenet", input_shape=IM_SIZE + (3,), include_top=False)
    base_cnn.trainable = False
    x = Flatten()(base_cnn.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(rate=drop)(x)
    x = Dense(256)(x)
    x = Dropout(rate=0.1)(x)
    output = Lambda(my_norm)(x)

    if finetune:
      trainable = False#default
      finetune_at =154
      count=0
      for layer in base_cnn.layers:
          if layer.name == "conv5_block1_1_conv":
              print("conv5_block found at count = ",count)
              trainable = True
          if count >finetune_at:
              #layer.trainable = True
              nothing = True
          layer.trainable = trainable
          count+= 1
      print("total no of layers: ", count)
    

    mdl = Model(inputs=base_cnn.input, outputs=output, name="Embedding")
    return mdl

def embedding_model_(inputhead, drop=0.3):
    # used inputhead which is the output of flatten
    #base_cnn = tf.keras.applications.ResNet50(weights="imagenet", input_shape=IM_SIZE + (3,), include_top=False)
    #x = Flatten()(base_cnn.output)
    x = Dense(256, activation="relu")(inputhead)
    x = Dropout(rate=drop)(x)
    x = Dense(256)(x)
    x = Dropout(rate=drop)(x)
    output = Lambda(my_norm)(x)
    
    return output

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
    return "resnet50tf1"


def preprocess(x):
    return tf.keras.applications.resnet50.preprocess_input(x)