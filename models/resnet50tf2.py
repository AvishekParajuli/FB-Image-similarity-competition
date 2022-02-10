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


def embedding_model1( drop=0.25, embeddingdim =256,finetune=True):
    # Simple convolutional model
    # used for the embedding model.
    base_cnn = tf.keras.applications.ResNet50(weights="imagenet", input_shape=IM_SIZE + (3,), include_top=False)
    #freze the base_cnn
    base_cnn.trainable = False
    flatten = Flatten()(base_cnn.output)
    drop1 = Dropout(rate=drop)(flatten)
    dense1 = Dense(256, activation="relu")(drop1)
    dense1 = BatchNormalization()(dense1)
    output = Dense(embeddingdim)(dense1)
    output = Lambda(my_norm)(output)
    
    #x = Conv2D(filters=512, kernel_size=1, strides=1, activation="relu")(x)
    #print("shape after new conv2d", x.shape)
    #x = GlobalAveragePooling2D()(x)
    
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

def embedding_model2(inputhead, drop =0.3, embeddingdim = 160):
    x = Conv2D(filters=96, kernel_size=8, strides=16,
     activation="relu",padding= "same", name ="em2_conv")(inputhead)
    x = MaxPool2D(pool_size = 3, strides = 4,padding="same")(x)
    x = Flatten()(x)
    #x = Dense(embeddingdim)(x)
    output = Lambda(my_norm)(x)
    #mdl = Model(inputs=input_head2, outputs=output, name="Embedding2")
    return output
    

def embedding_model3(inputhead,drop =0.3, embeddingdim = 160):
    #input_head3 = Input((imsize, imsize, 3))
    x = Conv2D(filters=96, kernel_size=8, strides=32,
     activation="relu",padding= "same",name ="em3_conv")(inputhead)
    x = MaxPool2D(pool_size = 7, strides = 2,padding="same")(x)
    x = Flatten()(x)
    #x = Dense(embeddingdim)(x)
    output = Lambda(my_norm)(x)
    return output
    
def embedding_model(embeddingdim=512, freezeFirst=True):
    input_0 = Input((imsize, imsize, 3))
    base1 = embedding_model1()
    print("before freeze")
    if freezeFirst:
        base1.trainable = False
    print("before base2")
    base2 = embedding_model2(input_0)
    print("before base3")
    base3 = embedding_model3(input_0)
    print("before 1st merge")
    merge1 = concatenate([base2, base3])
    print("before last concat")
    x = concatenate([base1.output, merge1]) #final merged
    print("shape after final concat", x.shape)
    x = Dense(embeddingdim)(x)
    output = Lambda(my_norm)(x)
    print("after output")
    mdl = Model(inputs=input_0, outputs=output, name="Embedding")
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

    loss = Lambda(triplet_loss)([A, P, N])
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(LR))
    return model


def get_model_name():
    return "resnet50tf2deep"


def preprocess(x):
    return tf.keras.applications.resnet50.preprocess_input(x)