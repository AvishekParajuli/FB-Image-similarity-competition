import keras
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D, Input, Lambda
from keras.utils import np_utils
from keras import backend as K
from settings import * # importing all the variables and Cosntants
from models.loss_funcs import *


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 0.8, lambda_param=5e-2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss1 = K.maximum(basic_loss,0.0)
    #add a regularization to minimize pos_dist
    loss2 = lambda_param*pos_dist
    loss = loss1+loss2
    return loss



