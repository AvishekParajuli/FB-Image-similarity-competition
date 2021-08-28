from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D, Input, Lambda
from keras.utils import np_utils
from keras import backend as K
from settings import * # importing all the variables and Cosntants

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def my_norm(ip):
    return K.l2_normalize(ip, axis=-1)

def embedding_model():
  # Simple convolutional model
  # used for the embedding model.
  model = Sequential()
  model.add(Conv2D(16, (3, 3), activation='relu',
                        input_shape=(imsize,imsize,3)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(64))
  return model


def complete_model(base_model):
    # Create the complete model with three
    # embedding models and minimize the loss
    # between their output embeddings
    input_1 = Input((imsize, imsize, 3))
    input_2 = Input((imsize, imsize, 3))
    input_3 = Input((imsize, imsize, 3))

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    A= Lambda(my_norm)(A)
    P = Lambda(my_norm)(P)
    N = Lambda(my_norm)(N)

    loss = Lambda(triplet_loss)([A, P, N])
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model

def get_model_name():
    return "baseline_simple"