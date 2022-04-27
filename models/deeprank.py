import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPool2D, MaxPooling2D, ZeroPadding2D, Input, Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate,BatchNormalization
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import backend as K
from settings import *  # importing all the variables and Cosntants
from models.loss_funcs import *

def my_norm(ip):
    return K.l2_normalize(ip, axis=-1)


def embedding_model1( base_cnn, drop=0.6, embeddingdim =4096,baseTrainable = False, finetune=False):
    # Simple convolutional model
    # used for the embedding model.
    #freze the base_cnn
    base_cnn.trainable = baseTrainable
    #x = Flatten()(base_cnn.output)
    x = GlobalAveragePooling2D()(base_cnn.output)
    x = Dense(embeddingdim, activation="relu")(x)
    x = Dropout(rate=drop)(x)
    x = Dense(embeddingdim, activation="relu")(x)
    x = Dropout(rate=drop)(x)
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
    #base_cnn.summary()
    mdl = Model(inputs=base_cnn.input, outputs=output, name="Embedding1")
    return mdl


def embedding_model2(inputhead):
    pad = "same"
    nf = 96
    x = Conv2D(filters=nf, kernel_size=8, strides=16,
     activation="relu",padding= pad, name ="em2_conv1")(inputhead)
    #print("x shape:", x.shape)
    x = MaxPool2D(pool_size = 3, strides = 4,padding=pad)(x)
    print("x shape:", x.shape)
    x = Flatten()(x)
    output = Lambda(my_norm)(x)
    return output
def embedding_model3(inputhead):
    pad = "same"
    nf = 96
    x = Conv2D(filters=nf, kernel_size=8, strides=32,
     activation="relu",padding= pad, name ="em3_conv1")(inputhead)
    print("x shape:", x.shape)
    x = MaxPool2D(pool_size = 7, strides = 2,padding=pad)(x)
    x = Flatten()(x)
    output = Lambda(my_norm)(x)
    return output


def embedding_model(drop=0.4, embeddingdim=4096, freezeFirstHead=False,basecnnTrainable = False, firstHeadWt=""):
    base_cnn = tf.keras.applications.ResNet50(weights="imagenet", input_shape=IM_SIZE + (3,), include_top=False)
    #freze the base_cnn
    base_cnn.trainable = basecnnTrainable
    input_0 = base_cnn.input

    base1 = embedding_model1(base_cnn, drop=drop)
    if firstHeadWt !='':
      base1.load_weights(firstHeadWt)
    if freezeFirstHead:
        base1.trainable = False
    
    base2 = embedding_model2(input_0)
    base3 = embedding_model3(input_0)

    merge1 = concatenate([base2, base3])
    x = concatenate([base1.output, merge1]) #final merged
    x = Dense(embeddingdim)(x)
    output = Lambda(my_norm)(x)
    
    mdl = Model(inputs=input_0, outputs=output, name="Embedding")
    return mdl

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

    loss = Lambda(triplet_loss)([A, P, N])
    print("not using triplet_loss_improved")
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model


def get_model_name():
    return "deeprank"


def preprocess(x):
    return tf.keras.applications.resnet50.preprocess_input(x)