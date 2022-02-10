import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
#from tensorflow.keras.layers import Conv2D, Dropout

def same_conv (inputs, nOutChannels, name,kernel_size = (3,3), strides = (1,1)):
    x = keras.layers.Conv2D(nOutChannels, kernel_size, strides, padding='same', activation='relu', name=name)(inputs)
    return x

def same_conv_wBN (inputs, nOutChannels, name,kernel_size = (3,3), strides = (1,1)):
    x = keras.layers.Conv2D(nOutChannels, kernel_size, strides=strides, padding='same', activation=None, name=name)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    return x

def my_norm(ip):
    return K.l2_normalize(ip, axis=-1)

def mixedStridesNet_wBN(inputSize, outChannels=2, nBaseFilters=8, dropout=0.25):
    nIpChan = int(inputSize.shape[-1]) #NHWC
    #input_img = keras.layers.Input(shape =(None, None, nIpChan),
    #                                  name="input_image",dtype=tf.float32,tensor=inputSize)#not including the batch size()
    input_img = keras.layers.Input(shape=inputSize, name="input_image")# for unit test model.summary()

    #now we same multiple same conv with mixed strides
    conv_2 = same_conv_wBN(input_img, 16, name='conv_2_1', strides=(4, 4), kernel_size=(4, 4))
    conv_2 = same_conv_wBN(conv_2, nBaseFilters * 1, name='conv_2_2')

    conv_4 = same_conv_wBN(conv_2, nBaseFilters * 2, name='conv_4_1', strides=(2, 2), kernel_size=(2, 2))
    conv_4 = same_conv_wBN(conv_4, nBaseFilters * 2, name='conv_4_2')

    conv_8 = same_conv_wBN(conv_4, nBaseFilters * 4, name='conv_8_1', strides=(2, 2), kernel_size=(2, 2))
    conv_8 = same_conv_wBN(conv_8, nBaseFilters * 4, name='conv_8_2')

    conv_16 = same_conv_wBN(conv_8, nBaseFilters * 8, name='conv_16_1', strides=(2, 2), kernel_size=(2, 2))
    conv_16 = same_conv_wBN(conv_16, nBaseFilters * 8, name='conv_16_2')

    net = tf.math.reduce_mean(conv_16, axis=[1, 2], keepdims=True, name='global_pool')
    net = keras.layers.Dropout(rate=dropout)(net)

    net = keras.layers.Conv2D(outChannels, kernel_size=(1, 1), activation=None, name='conv_last')(net)
    #net = squeeze(net)##net = Lambda(squeeze)(net)--> lambda is not required
    net = keras.layers.Flatten()(net)

    model = keras.Model(inputs=input_img, outputs= net)
    return model

def residual_module(inputLayer, nOutFilters, ksize=(3,3), strides =(2,2), name="res"):
    resd = inputLayer


    x = keras.layers.Conv2D(nOutFilters, kernel_size=ksize,padding='same',strides=strides,
                               name = name+"_conv1" )(inputLayer)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(nOutFilters, kernel_size=ksize, padding='same',#default strides=(1,1)
                               name=name + "_conv2")(x)
    #add with the residue layer
    #print("x.shape", x.shape)
    #print("resd.shape", resd.shape)
    # check no of filters of input and output layers and if they aren't same
    #also check length/width
    stride_init = (1,1)
    str_h = resd.shape[1] // x.shape[1]
    #print("====determine stride:", str_h)
    if str_h>1:
        #print(resd.shape[1],x.shape[1])
        # print("====determine stride:", str_h)
        stride_init = (str_h, str_h)#assume Height and Width are equal

    if inputLayer.shape[-1] != nOutFilters or str_h>1:
        resd = keras.layers.Conv2D(nOutFilters, kernel_size=(1, 1),strides=stride_init, padding='same',
                                      activation='relu')(inputLayer)
    #print("resd.shape after stride change", resd.shape)
    merged = keras.layers.add([resd, x])
    #keras.layers.Add()([resd, x]) equivalent to line above
    return keras.layers.Activation('relu')(merged)


def resnet18_vanilla(inputSize, outChannels=256, nBaseFilters=32, dropout=0.25):
    #nIpChan = inputSize[-1]# enable this for testing and disable below
    #nIpChan = int(inputSize.shape[-1]) #enable this for actula model NHWC
    #input_img = keras.layers.Input(shape =(224,224, nIpChan),name="input_image",dtype=tf.float32
    #                                  ,tensor=inputSize)#not including the batch size()
    input_img = keras.layers.Input(shape=inputSize, name="input_image")# for unit test model.summary()

    #now we call resnet-18
    #inputsize=224x224
    x1 = same_conv_wBN(input_img,nBaseFilters , name='block1', strides=(2,2), kernel_size=(7,7))
    x1 = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(x1)
    #print("======x1.shape", x1.shape)
    #output: 112x112

    x2 = residual_module(x1,nBaseFilters,ksize=(3,3),strides=(1,1), name="block2" )
    x2 = residual_module(x2, nBaseFilters, ksize=(3, 3),strides=(1,1), name="block2_1")
    #print("=====x2.shape", x2.shape)
    #output_size: 56x56

    x3 = residual_module(x2, nBaseFilters * 2, ksize=(3, 3), name="block3")
    x3 = residual_module(x3, nBaseFilters * 2, ksize=(3, 3),strides=(1,1), name="block3_1")
    #print("========x3shape", x3.shape)
    #output_size:28x28

    x4 = residual_module(x3, nBaseFilters * 4, ksize=(3, 3), name="block4")
    x4 = residual_module(x4, nBaseFilters * 4, ksize=(3, 3),strides=(1,1), name="block4_1")
    #print("=======x4shape", x4.shape)
    #output_size:14x14

    x5 = residual_module(x4, nBaseFilters * 8, ksize=(3, 3), name="block5")
    x5 = residual_module(x5, nBaseFilters * 8, ksize=(3, 3),strides=(1,1), name="block5_1")
    #print("=========x5shape", x5.shape)
    # output_size:7x7


    #apply average pooling
    x = keras.layers.GlobalAveragePooling2D()(x5)
    x = keras.layers.Dropout(rate=dropout)(x)
    # output_size:1x1
    #print("========shape after globalAverage", x.shape)

    x = keras.layers.Dense(outChannels,activation=None, name="last_dense")(x)
    x = keras.layers.Lambda(my_norm)(x)
    #x = keras.layers.Dense(outChannels, activation="sigmoid", name="last_dense")(x)
    #print("=========shape after dense", x.shape)
    #net = tf.squeeze(x, axis=-1)  ##net = Lambda(squeeze)(net)--> lambda is not required

    model = keras.Model(inputs=input_img, outputs=x)

    return model

def resnet50():
    base_model = keras.applications.resnet50.ResNet50()
def mobilenetv2():
    base_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet")
    return base_model

