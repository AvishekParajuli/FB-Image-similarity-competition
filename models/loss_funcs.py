import keras
from keras import backend as K
from settings import *  # importing all the variables and Cosntants

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(x, alpha=ALPHA, lambda_param =5e-2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss1 = K.maximum(basic_loss, 0.0)
    loss2 = lambda_param*pos_dist
    loss = loss1+loss2
    return loss


