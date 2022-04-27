import keras
from keras import backend as K
from settings import *  # importing all the variables and Cosntants

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(x, alpha=ALPHA, lambda_param =1e-3):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss1 = K.maximum(basic_loss, 0.0)
    #loss2 = lambda_param*pos_dist
    #loss = loss1+loss2
    return loss1 #K.mean(loss)

def triplet_loss_improved(x, alpha=ALPHA, lambda_param =1e-3):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    neg_dist2 = K.sum(K.square(positive - negative), axis=1)
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss1 = K.maximum(basic_loss, 0.0)
    loss2 = K.maximum(0.0, pos_dist - neg_dist2 + 0.5*alpha)
    loss = loss1+loss2
    return loss #K.mean(loss)


def triplet_loss_softplus(x, alpha=ALPHA, lambda_param =5e-2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    # compute loss
    alpha = 1.0
    basic_loss = pos_dist - neg_dist
    loss = K.log( alpha + K.exp(basic_loss))
    return K.mean(loss)




