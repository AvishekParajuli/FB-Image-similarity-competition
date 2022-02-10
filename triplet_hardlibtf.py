''' Simple triplet based similarity training;
Simple: using a simple model with dense layers and few conv2d only
ref:https://github.com/Ekeany/Siamese-Network-with-Triplet-Loss/blob/master/MachinePart1.ipynb
#requires keras 2.2.5(cityscape env)
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from args import get_arguments
#%load_ext autoreload
#%autoreload 2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path
from settings import * # importing all the variables and Cosntants
from getmodel import *

import tensorflow
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard
from importlib import reload
import loader
reload (loader)
#from loader.fb_image_gen import *
from loader.fb_image_gen_pre_tf import *
import pickle
from datetime import datetime
import time
import faiss
from utils import *




def getArgOptions():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    '''
    group = parser.add_argument_group('feature extraction options')
    aa('--transpose', default=-1, type=int, help="one of the 7 PIL transpose options ")
    aa('--train_pca', default=False, action="store_true", help="run PCA training")
    aa('--pca_file', default="", help="File with PCA descriptors")
    aa('--pca_dim', default=256, type=int, help="output dimension for PCA")
    aa('--pca_white', default=0.0, type=float, help="set to -0.5 to whiten PCA")
    

    group = parser.add_argument_group('dataset options')
    
    aa('--anchor_file_list', default='./list_files/subset_1_queries', help="CSV file with query image filenames")
    aa('--anchor_img_dir', default="D:/prjs/im-similarity/data/query", help="search image files in this directory")
    aa('--ref_file_list', default='./list_files/subset_1_references', help="CSV file with reference imagenames")
    aa('--ref_img_dir', default="D:/prjs/im-similarity/data/reference", help="search image files in this directory")
    
    aa('--n_train_pca', default=10000, type=int, help="nb of training vectors for the PCA")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")
    '''

    group = parser.add_argument_group('output options')
    aa('--o', default="./desc.hdf5", help="write trained features to this file")

    args = parser.parse_args()
    print("args=", args)

    #print("reading anchor image names from", args.anchor_file_list)
    #print("reading ref image names from", args.ref_file_list)
    return args
def print_date():
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_H%H_M%M")
    print("current date time: ",dt_string)
    
def train_basic(model, base_model, epochs=10,batchsize = 32):
    image_count= 4976# this si actual length of data in seq--------- len(Q_List)
    train_stop_idx = 0.9*image_count
    bs=batchsize

    #train_generator = generate_triplets(start=0,stop=train_stop_idx,BATCH_SIZE=bs,mode='train')
    #test_generator = generate_triplets(start=train_stop_idx+1, stop=image_count-1, BATCH_SIZE=bs)
    train_generator = generate_triplets_hdfseq(start=0,stop=train_stop_idx, batch_sz=bs)
    test_generator = generate_triplets_hdfseq(start=train_stop_idx+1, stop=image_count-1, batch_sz=bs)
    data = next(train_generator)
    #plot_triplets(data)
    EPOCHS = epochs


    #base_model = embedding_model()
    #model = complete_model(base_model)
    #model.summary()
    modelFilePath = "./models/weights/"
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_H%H_M%M")
    print_date()
    model_save_name = modelFilePath+"model_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    embeddings_save_name = modelFilePath+"em_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    #print("model weights filepath name is: ",model_save_name)

    stps= train_stop_idx//bs
    my_callbacks =[EarlyStopping(patience=PATIENCE),
                   ModelCheckpoint(filepath=modelFilePath+'model.h5',
                                   save_weights_only=True,save_best_only=True,monitor='val_loss'),
                   TensorBoard(log_dir='./models/logs')]
    valid_Stps = (image_count-train_stop_idx)//bs
    history = model.fit_generator(train_generator,epochs=EPOCHS, steps_per_epoch=stps,
                                  validation_data=test_generator, validation_steps=valid_Stps, callbacks=my_callbacks, verbose=1)



    #model.save_weights(model_save_name)
    #base_model.save_weights(embeddings_save_name)


    testgen = generate_triplets(BATCH_SIZE=1)
    adata = next(testgen )

    pred = model.predict(adata[0])
    a_em = np.squeeze(base_model.predict(adata[0][0]))
    b_em = np.squeeze(base_model.predict(adata[0][1]))
    c_em = np.squeeze(base_model.predict(adata[0][2]))

    #this sum(square) metric is better
    print("sum(square)difference between anchor and Positive:", np.sum(np.square(a_em - b_em)))
    print("sum(square)difference between anchor and negative:", np.sum(np.square(a_em - c_em)))
    
    return model, base_model

def train_basic_traindev(model, base_model, epochs=10,batchsize = 32):
    image_count= 50000 #100_00# this si actual length of data in seq--------- len(Q_List)
    train_stop_idx = 1.0*image_count
    bs=batchsize
    
    print_date()

    train_generator = generate_triplets_train_hdfseq(start=0, stop=train_stop_idx, batch_sz=bs)
    test_generator = generate_triplets_hdfseq( batch_sz=bs)
    data = next(train_generator)
    #plot_triplets(data)
    EPOCHS = epochs

    modelFilePath = "./models/weights/"

    stps= train_stop_idx//bs
    my_callbacks =[EarlyStopping(patience=20),
                   ModelCheckpoint(filepath=modelFilePath+'model_traindev.hdf5',
                                   save_weights_only=True,save_best_only=True,monitor='val_loss'),
                   TensorBoard(log_dir='./models/logs')]
    valid_Stps = (4991)//bs
    print("Starting train dev training process!")
    print("-------------------------------------")
    history = model.fit_generator(train_generator,epochs=EPOCHS, steps_per_epoch=stps,
                                  validation_data=test_generator, validation_steps=valid_Stps, callbacks=my_callbacks,
                                  verbose=1)

    testgen = generate_triplets(BATCH_SIZE=1)
    adata = next(testgen )

    pred = model.predict(adata[0])
    a_em = np.squeeze(base_model.predict(adata[0][0]))
    b_em = np.squeeze(base_model.predict(adata[0][1]))
    c_em = np.squeeze(base_model.predict(adata[0][2]))

    #this sum(square) metric is better
    print("sum(square)difference between anchor and Positive:", np.sum(np.square(a_em - b_em)))
    print("sum(square)difference between anchor and negative:", np.sum(np.square(a_em - c_em)))
    return model, base_model

def test_hardbatch(model, base_model, epochs, batchsize = 32, largeBS = 100):
    image_count = len(Q_List)
    train_stop_idx = int(0.8 * image_count)

    largeBS = largeBS
    bs = batchsize
    usehdf5Sequence = True
    patience = PATIENCE

    #base_model = embedding_model()
    #triplets, labels = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)# if return
    #hardbatch_gen = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)

    large_Generator = generate_triplets(start=0, stop=train_stop_idx, BATCH_SIZE=largeBS,mode ='train')
    test_generator = generate_triplets(start=train_stop_idx+1, stop=image_count-1, BATCH_SIZE=bs)
    if usehdf5Sequence:
        large_Generator = generate_triplets_hdfseq(start=0, stop=train_stop_idx, batch_sz=largeBS)
        test_generator = generate_triplets_hdfseq(start=train_stop_idx + 1, stop=image_count - 1, batch_sz=bs)
    #model = complete_model(base_model)
    #model.compile(loss=identity_loss, optimizer=Adam(1e-4))
    import time
    from datetime import datetime

    modelFilePath = "./models/weights/"
    #base_model.load_weights(modelFilePath + "Embeddings_best.hdf5")
    #model.load_weights(modelFilePath + "complete_res18_best.hdf5")
    EPOCHS = epochs
    print_date()

    #history = model.fit_generator(hardbatch_gen, epochs=2, steps_per_epoch=10,
    #                              validation_data=test_generator, validation_steps=10)

    steps_per_ep = int(train_stop_idx//bs)
    steps_per_eval = int((image_count-train_stop_idx)//bs)
    n_iter = steps_per_ep*EPOCHS
    n_iteration=0#starting count
    best_val_loss = 1000
    eval_every = min(100,steps_per_ep)
    best_val_index = 0

    print("Starting Semi-Hard Negative training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        #triplets,labels = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)
        triplets,labels = get_batch_semihardNeg(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs, alpha=ALPHA)
        loss = model.train_on_batch(triplets, labels)
        n_iteration += 1
        if i % eval_every == 0:
            print("{}/{} -------------".format(i,n_iter))
            print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time() - t_start) / 60.0,
                                                                                      loss, n_iteration))
            val_loss = []
            for ii in range(steps_per_eval):
                data, labels = next(test_generator)
                val_loss.append(model.predict_on_batch(data))
            curr_val_loss = np.mean(np.mean(val_loss))
            print("val_loss = ", curr_val_loss)
            if(curr_val_loss <best_val_loss):
                print("best loss found, previous: {}, current: {} ".format(best_val_loss,curr_val_loss))
                best_val_loss = curr_val_loss
                best_val_index = i
                print("curr best_val_index= ", best_val_index)
                base_model.save_weights(modelFilePath + "SMHD_Embeddings_best.hdf5")
                #model.save_weights(modelFilePath + "complete_res18_best.hdf5")
        if ((n_iteration - best_val_index) > patience * steps_per_ep):
            print("best val loss={}, at iter={}".format(best_val_loss, best_val_index))
            break

            #probs, yprob = compute_probs(network, x_test_origin[:n_val, :, :, :], y_test_origin[:n_val])
    #model.save_weights(model_save_name)
    #base_model.save_weights(embeddings_save_name)
    return model, base_model


def test_hardbatch_traindev(model, base_model, epochs, batchsize = 20, largeBS = 100):
    image_count = len(Q_List)#used for test
    train_count = 50000
    largeBS = largeBS
    bs = batchsize
    patience = PATIENCE
    EPOCHS = epochs

    large_Generator = generate_triplets_train_hdfseq(start=0, stop=train_count, batch_sz=largeBS, forcePrep = True)
    test_generator = generate_triplets_hdfseq(start=0, stop=image_count-1, batch_sz=64)
    
    modelFilePath = "./models/weights/"
    
    print_date()

    steps_per_ep = int(train_count//bs)
    steps_per_eval = int((image_count)//bs)
    n_iter = steps_per_ep*EPOCHS
    n_iteration=0#starting count
    best_val_loss = 1000
    eval_every = min(1000,steps_per_ep)
    best_val_index = 0

    print("Starting Semi-Hard Negative training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        #triplets,labels = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)
        triplets,labels = get_batch_semihardNeg(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs, alpha=ALPHA, hard_perct =0.5)
        loss = model.train_on_batch(triplets, labels)
        n_iteration += 1
        if i % eval_every == 0:
            print("{}/{} -------------".format(i,n_iter))
            print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time() - t_start) / 60.0,
                                                                                      loss, n_iteration))
            val_loss = []
            for ii in range(steps_per_eval):
                data, labels = next(test_generator)
                val_loss.append(model.predict_on_batch(data))
            curr_val_loss = np.mean(np.mean(val_loss))
            print("val_loss = ", curr_val_loss)
            if(curr_val_loss <best_val_loss):
                print("best loss found, previous: {}, current: {} ".format(best_val_loss,curr_val_loss))
                best_val_loss = curr_val_loss
                best_val_index = i
                print("curr best_val_index= ", best_val_index)
                base_model.save_weights(modelFilePath + "SMHD_traindev_base.hdf5")
                #model.save_weights(modelFilePath + "complete_res18_best.hdf5")
        if ((n_iteration - best_val_index) > patience * steps_per_ep):
            print("best val loss={}, at iter={}".format(best_val_loss, best_val_index))
            break

            #probs, yprob = compute_probs(network, x_test_origin[:n_val, :, :, :], y_test_origin[:n_val])
    #model.save_weights(model_save_name)
    #base_model.save_weights(embeddings_save_name)
    return model, base_model

def getHardNegList(I, k =1):
  out =[]
  lenI= len(I)
  if k==1:
    matches = [index for index, value in enumerate(I) if value[0]==index]
  else:
    matches = [index for index, value in enumerate(I) if index in value[0:k]]
  #non_matching_idx = [i for i in range(lenI) if i not in matches]
  non_matching_idx = [i for i in range(lenI) if i not in matches]
  negIdx = [I[ii][0] for ii in non_matching_idx]
  return non_matching_idx, negIdx
def combine_Is(Icurr, Iprev):
  queryId1, negId1 = getHardNegList(Icurr)
  queryId2, negId2 = getHardNegList(Iprev)
  Inew = Icurr.copy()
  diffIndex = [ idx  for idx,val in enumerate(queryId2) if val not in queryId1]
  #print("diff index ={}".format(diffIndex))
  newqryId = [ queryId2[id] for id in diffIndex]
  #print("newqrryId =", newqryId)
  cnt=0
  for idx, val in enumerate(newqryId):
    Inew[val] = negId2[diffIndex[idx]]
  return Inew

def test_hardOfflineBatch(model, base_model, epochs,batchsize = 16):
    Isaved = pickle.load(open("./data/L2Index_2_prev.p", "rb"))
    queryId, negId = getHardNegList(Isaved)

    count = len(queryId)
    train_stop = int(0.8 * count)
    print("train_stop= ", train_stop)
    bs = batchsize
    EPOCHS = epochs #10
    patience = PATIENCE  # in epochs
    train_generator = generate_offline_triplets(queryId, negId, 0, count, BATCH_SIZE=bs)
    test_generator = generate_offline_triplets(queryId, negId, train_stop + 1, count, BATCH_SIZE=bs)
    print_date()
    #model = complete_model(base_model)
    #model.compile(loss=identity_loss, optimizer=Adam(1e-4))

    modelFilePath = "./models/weights/"
    #base_model.load_weights(modelFilePath + "Embeddings_best.hdf5")
    #model.load_weights(modelFilePath + "complete_res18_best.hdf5")
    

    #history = model.fit_generator(hardbatch_gen, epochs=2, steps_per_epoch=10,
    #                              validation_data=test_generator, validation_steps=10)

    steps_per_ep = int(train_stop//bs)
    n_iter = steps_per_ep*EPOCHS
    n_iteration=0#starting count
    best_val_loss = 1000
    eval_steps = int((count - train_stop)//bs)+2
    eval_every_nsteps = 100
    best_val_index = 0

    print("Starting HardOffline training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        triplets,labels = next(train_generator)
        loss = model.train_on_batch(triplets, labels)
        n_iteration += 1
        if i % eval_every_nsteps == 0:
            print("{}/{} -------------".format(i,n_iter))
            print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time() - t_start) / 60.0,
                                                                                      loss, n_iteration))
            val_loss = []
            for ii in range(eval_steps):
                data, labels1 = next(test_generator)
                val_loss.append(model.predict_on_batch(data))
            curr_val_loss = np.mean(np.mean(val_loss))
            print("val_loss = ", curr_val_loss)
            if(curr_val_loss <best_val_loss):
                print("best loss found, previous: {}, current: {} ".format(best_val_loss,curr_val_loss))
                best_val_loss = curr_val_loss
                best_val_index = i
                print("curr best_val_index= ", best_val_index)
                base_model.save_weights(modelFilePath + "OFF_Embeddings_res50_best.hdf5")
                #model.save_weights(modelFilePath + "complete_res50_best.hdf5")
        if ((n_iteration - best_val_index) > patience * steps_per_ep):
            print("best val loss={}, at iter={}".format(best_val_loss, best_val_index))
            break
    #model.save_weights(modelFilePath + "complete_final.hdf5")
    #base_model.save_weights(modelFilePath + "Embeddings_final.hdf5")
    return model, base_model
def merge_triplets(triplets1,triplets2):
  a1,p1,n1 = triplets1
  a2,p2,n2 = triplets2
  A = np.vstack((a1,a2))
  P = np.vstack((p1,p2))
  N = np.vstack((n1,n2))
  return [A, P, N]

def test_hardOffline_online_Batch(model, base_model, epochs,batchsize = 16,pat=100):
    Isaved = pickle.load(open("./data/L2Index_2_prev.p", "rb"))
    queryId, negId = getHardNegList(Isaved)

    count = len(queryId)
    train_stop = int(0.8 * count)
    print("train_stop= ", train_stop)
    bs = batchsize
    EPOCHS = epochs #10
    patience = pat  # in epochs
    large_Generator = generate_triplets_train_hdfseq(start=0, stop=50000, batch_sz=100, forcePrep = True)
    
    train_generator = generate_offline_triplets(queryId, negId, 0, count, BATCH_SIZE=bs//2)
    test_generator = generate_offline_triplets(queryId, negId, train_stop + 1, count, BATCH_SIZE=bs)
    print_date()
    
    modelFilePath = "./models/weights/"

    steps_per_ep = int(count//bs)
    n_iter = steps_per_ep*EPOCHS
    n_iteration=0#starting count
    best_val_loss = 1000
    eval_steps = int((count - train_stop)//bs)+2
    eval_every_nsteps = 300
    best_val_index = 0

    print("Starting HardOffline training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        triplets1,labels1 = next(train_generator)
        triplets2,labels2 = get_batch_semihardNeg(base_model, large_Generator, draw_batch_size=100,actual_batch_size=bs//2, alpha=ALPHA, hard_perct =0.5)
        #triplets = np.vstack((triplets1,triplets2))
        triplets = merge_triplets(triplets1,triplets2)
        labels = np.ones((bs,1))
        #labels = np.vstack((labels1,labels2))
        if i<2:
            print("triplets1.shape={},tripelet2 ={},merged={}".format(triplets1[0].shape,triplets2[0].shape,triplets[0].shape))
        try:
          loss = model.train_on_batch(triplets, labels)
        except Exception as e:
          print("triplets1.shape={},tripelet2 ={},merged={}".format(triplets1[0].shape,triplets2[0].shape,triplets[0].shape))
          print("********ERROR: labels shape",labels.shape)
          print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        n_iteration += 1
        if i % eval_every_nsteps == 0:
            print("{}/{} -------------".format(i,n_iter))
            print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time() - t_start) / 60.0,
                                                                                      loss, n_iteration))
            val_loss = []
            for ii in range(eval_steps):
                data, labels0 = next(test_generator)
                val_loss.append(model.predict_on_batch(data))
            curr_val_loss = np.mean(np.mean(val_loss))
            print("val_loss = ", curr_val_loss)
            if(curr_val_loss <best_val_loss):
                print("best loss found, previous: {}, current: {} ".format(best_val_loss,curr_val_loss))
                best_val_loss = curr_val_loss
                best_val_index = i
                print("curr best_val_index= ", best_val_index)
                base_model.save_weights(modelFilePath + "OFF_Embeddings_res50_best.hdf5")
                #model.save_weights(modelFilePath + "complete_res50_best.hdf5")
        if ((n_iteration - best_val_index) > patience * steps_per_ep):
            print("best val loss={}, at iter={}".format(best_val_loss, best_val_index))
            break
    #model.save_weights(modelFilePath + "complete_final.hdf5")
    #base_model.save_weights(modelFilePath + "Embeddings_final.hdf5")
    return model, base_model

def trainLoop():
    base_model = embedding_model()
    model = complete_model(base_model)
    model.summary()
    model.compile(loss=identity_loss, optimizer=Adam(1e-4))
    modelFilePath = "./models/weights/"
    model.load_weights(modelFilePath + "resnet50Reg0.8complete_final.hdf5")
    #model.load_weights(modelFilePath + "model.h5")
    model, base_model = test_hardbatch(model, base_model, epochs=1,batchsize=32,largeBS=64)
    model, base_model = train_basic(model, base_model, epochs=2)#40
    Ibasic = findAccuracy(base_model)
    pickle.dump(Ibasic ,open("./data/L2Index_2_prev.p", "wb"))
    model, base_model = test_hardOfflineBatch(model, base_model, epochs=2)#orig=20
    #model.load_weights(modelFilePath+"resnet50Regcomplete_90%.hdf5")
    #base_model.load_weights(modelFilePath + "Embeddings_res50_best.hdf5")
    Ioff = findAccuracy(base_model)
    pickle.dump(Ioff, open("./data/L2Index_2_prev.p", "wb"))
    model, base_model = test_hardbatch(model, base_model, epochs=1)
    Ihard = findAccuracy(base_model)
    pickle.dump(Ihard, open("./data/L2Index_2_prev.p", "wb"))
    model, base_model = test_hardOfflineBatch(model, base_model, epochs=10)
    findAccuracy(base_model)
    model.save_weights(modelFilePath +get_model_name()+ "complete_final.hdf5")
    base_model.save_weights(modelFilePath + get_model_name()+"Embeddings_final.hdf5")

def generate_subset_embeddings():
    base_model = embedding_model()
    base_model.load_weights("./models/weights/" + "resnet50Reg0.8Embeddings_final.hdf5")
    #base_model.load_weights("./models/weights/" + "OFF_Em_res50_best.hdf5")
    #base_model.load_weights("./models/weights/" + "resnet50Regbase_90_.hdf5")
    #findAccuracy(base_model, save=False) #uses old method
    image_list, ids = getImIds('./list_files/subset_1_queries',"D:/prjs/im-similarity/data/query")
    XQ = gen_embeddingsSeq(base_model, './data/image/im_subset_query.hdf5', ids,
                           outFile='./data/embed/subset_query_em_resnet50Reg.hdf5',batch=50)
    image_list, ids = getImIds('./list_files/subset_1_references',
                               "C:/Users/parajav/PycharmProjects/isc/reference/reference")
    XD = gen_embeddingsSeq(base_model,'./data/image/im_subset_ref.hdf5', ids,
                           outFile='./data/embed/subset_ref_em_resnet50Reg.hdf5', batch=50)
    d = XD.shape[-1]#EM_DIM
    index = faiss.IndexFlatL2(d)
    k = 1
    index.add(XD)
    D, I = index.search(XQ, k)  # search top k
    print("matching index after training....")
    # print(I)
    getMatchingScore(I, k)

def getClassifierMetrics(y_test, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_score, \
        recall_score, precision_recall_curve
    from sklearn.metrics import f1_score, balanced_accuracy_score
    print("*************Suggested accuracy from metrics evaluation************")
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
    print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
    print(f'Recall score: {recall_score(y_test, y_pred)}')


def getBestThreshold(probas_pred, y_true):

    from sklearn.linear_model import LogisticRegression
    X = probas_pred.reshape(-1, 1)
    #***********weighted logistic classifier****************
    w = {False: 1, True: 20}# change from 7 to 7.5 to see the effect(def:7.05)
    # define model
    lg2 = LogisticRegression(random_state=13, class_weight=w)
    # fit it
    lg2.fit(X, y_true)
    y_pred = lg2.predict(X)
    getClassifierMetrics(y_true, y_pred)
    truepred_dist = X[y_pred == True]
    falsepred_dist = X[y_pred == False]
    outThresh = min(truepred_dist)
    print(" Suggested threshold ", outThresh)
    print(" Closest other class dist: ", max(falsepred_dist))
    return outThresh

def get_optimizedmetrics(XQ, ids, XD, rids,outfileSuf='_', submission=False):
    from isc.metrics import to_arrays
    from isc.metrics import evaluate, print_metrics
    from isc.io import read_ground_truth, read_descriptors, write_predictions
    from isc.descriptor_matching import knn_match_and_make_predictions
    from scripts.compute_metrics import plot_pr_curve
    print("XQ embedding array shape:", XQ.shape)
    print("XD embedding array shape:", XD.shape)
    
    predictions = knn_match_and_make_predictions(XQ[0:25000], ids[0:25000], XD, rids, 1, metric=faiss.METRIC_L2)

    # ids, XQ = read_descriptors(['./data/embed/full_query_em_resnet50Reg.hdf5'])
    gt_matches = read_ground_truth('./list_files/subset_1_ground_truth.csv')
    metrics = evaluate(gt_matches, predictions)
    print_metrics(metrics)
    plot_pr_curve(metrics,"pr-curve-25k", "pr-curve25k.jpg")

    y_true, probas_pred = to_arrays(gt_matches, predictions)
    print("*******Total no of correct predictions: ", len(probas_pred[y_true]))
    print("*******Total no of incorrect predictions: ", len(probas_pred[y_true == False]))
    bestThresh = getBestThreshold(probas_pred, y_true)[0]
    bestThresh = -bestThresh #negate
    if submission:
        predictions = knn_match_and_make_predictions(XQ, ids, XD, rids, 1, metric=faiss.METRIC_L2)
        print("writing predictions to", './data/fullQ_Ref' + outfileSuf + '_WoTH_submit.csv')
        write_predictions(predictions, './data/fullQ_Ref' + outfileSuf + '_WoTH_submit.csv')
        y_true, probas_pred = to_arrays(gt_matches, predictions)
        metrics = evaluate(gt_matches, predictions)
        print_metrics(metrics)
        plot_pr_curve(metrics,"pr-curve-50k", "pr-curve50k.jpg")
        with open('./list_files/fullQ_Ref_mined_negsid_WoTH' + outfileSuf + '.csv', "w") as pfile:
            pfile.write("query_id,neg_id,score\n")
            for count, p in enumerate(predictions):
                if (y_true[count] == False):
                    row = f"{p.query},{p.db},{p.score}"
                    pfile.write(row + "\n")
                count += 1
        del predictions
        predictions = knn_match_and_make_predictions(XQ, ids, XD, rids, 1, metric=faiss.METRIC_L2, DIST_TH=bestThresh)
        print("writing predictions to", './data/fullQ_Ref' + outfileSuf + '_WTH_submit.csv')
        write_predictions(predictions, './data/fullQ_Ref' + outfileSuf + '_WTH_submit.csv')
    else:
        print("writing predictions to", './data/25kQ_Ref' + outfileSuf + '_raw.csv')
        write_predictions(predictions, './data/25kQ_Ref' + outfileSuf + '_raw.csv')
        with open('./list_files/25kQ_Ref_mined_negsid' + outfileSuf + '.csv', "w") as pfile:
            pfile.write("query_id,neg_id,score\n")
            for count, p in enumerate(predictions):
                if (y_true[count] == False):
                    row = f"{p.query},{p.db},{p.score}"
                    pfile.write(row + "\n")
                count += 1
    print_date()
    del XQ, XD

def generate_full_QueryEmbeddings(base_model='', base_model_filename='', outfileSuf ='_'):
    if base_model =='':
        #load model weights only if base_model is empty
        base_model = embedding_model()
        #base_model.load_weights("./models/weights/" + "resnet50goodEmbeddings_86_.hdf5")
        #base_model.load_weights("./models/weights/" + "OFF_Em_res50_best.hdf5")
        if base_model_filename == '':
            base_model.load_weights("./models/weights/" + "OFF_Em_res50_best.hdf5")
        else:
            base_model.load_weights(base_model_filename)
    name = get_model_name()
    img_dir = "D:/prjs/im-similarity/data/query"#path doesn't matter, we are only using the ids
    _rimage_list, rids = getImIds('./list_files/subset_ref_extended',
                                 "C:/Users/parajav/PycharmProjects/isc/reference/reference")
    XD = gen_embeddingsSeq(base_model, './data/image/image_extended_Ref.hdf5', rids,
                           outFile='./data/embed/subset_refExtended_em_'+name+'.hdf5', batch=50)
    # img_dir = 'C:/Users/parajav/PycharmProjects/isc/query' './list_files/subset_1_queries'
    _image_list, ids = getImIds('./list_files/dev_queries', img_dir)  # './list_files/dev_queries'
    XQ =gen_embeddingsSeq(base_model, './data/image/image_dev_queries.hdf5',ids[0:25000],
                          outFile='./data/embed/full_query_em_'+name+'.hdf5', batch=50)
    get_optimizedmetrics(XQ, ids[0:25000], XD, rids, outfileSuf)
    del XQ, XD
    
def generate_full_RefEmbeddings(base_model='', base_model_filename='',gen_embed=True):
    from isc.io import read_descriptors
    if base_model =='':
        #load model weights only if base_model is empty
        base_model = embedding_model()
        #base_model.load_weights("./models/weights/" + "resnet50goodEmbeddings_86_.hdf5")
        #base_model.load_weights("./models/weights/" + "OFF_Em_res50_best.hdf5")
        if base_model_filename == '':
            base_model.load_weights("./models/weights/" + "OFF_Em_res50_best.hdf5")
        else:
            base_model.load_weights(base_model_filename)
    
    name = get_model_name()
    ref_file_list = './list_files/references'
    ref_img_dir = 'C:/Users/parajav/PycharmProjects/isc/reference/reference'
    ref_image_list, ref_ids = getImIds(ref_file_list, ref_img_dir)
    XD =[]
    XQ =[]
    print("Total IDS:", len(ref_ids))
    if gen_embed==True:
      interval = 50000#50K
      iters = int(len(ref_ids)/interval)
      hdf5_imagelist = ['./data/image/image_full_ref_' + str(i) + '.hdf5' for i in range(20)]
      print("iters", iters)
      for ii in range(0,iters):
          i0 =ii*interval
          i1 = (ii+1)*interval
          print("reading files from {} to {}".format(i0, i1))
          XD =gen_embeddingsSeq(base_model, hdf5_imagelist[ii], file_ids=ref_ids[i0:i1],
                          outFile='./data/embed/full_ref_em_' +str(ii)+ name + '.hdf5',batch=100)
          del XD
    
    anchor_img_dir = "D:/prjs/im-similarity/data/query"
    q_image_list, q_ids = getImIds('./list_files/dev_queries', anchor_img_dir)
    if gen_embed==True:
      XQ =gen_embeddingsSeq(base_model, './data/image/image_dev_queries.hdf5',q_ids,
                          outFile='./data/embed/full_query_em_'+name+'.hdf5', batch=50)
    else:
      q_image_ids, XQ = read_descriptors(['./data/embed/full_query_em_'+name+'.hdf5'])
    
    db_descs = ['./data/embed/full_ref_em_' + str(i) + name + '.hdf5' for i in range(20)]
    db_image_ids, XD = read_descriptors(db_descs)
    get_optimizedmetrics(XQ, q_ids, XD, ref_ids, 'final',submission=True)

def train_public_gt(epochs=20, bs=32):
    print_date()
    base_model = embedding_model()
    model = complete_model(base_model)
    model.load_weights("./models/weights/" + "resnet50Reg0.8complete_final.hdf5")
    model.summary()
    model.compile(loss=identity_loss, optimizer=Adam(1e-4))
    #base_model.load_weights("./models/weights/" + "resnet50Regbase_90_.hdf5")
    name = get_model_name()
    filename = './list_files/public_ground_truth.csv'
    public_gt_array = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == "query_id,reference_id":
                continue
            q, db = line.split(",")
            public_gt_array.append([q,db])
    count = len(public_gt_array)
    train_stop = int(0.8 * count)
    print("train_stop= ", train_stop)
    EPOCHS = epochs  # 10
    patience = PATIENCE  # in epochs
    train_generator = generate_dev_triplets(public_gt_array[0:train_stop], batch_sz=bs)
    test_generator = generate_dev_triplets(public_gt_array[train_stop:count], batch_sz=bs)

    #model.compile(loss=identity_loss, optimizer=Adam(1e-4))

    modelFilePath = "./models/weights/"
    # base_model.load_weights(modelFilePath + "Embeddings_best.hdf5")
    # model.load_weights(modelFilePath + "complete_res18_best.hdf5")

    steps_per_ep = int(train_stop // bs)
    n_iter = steps_per_ep * EPOCHS
    n_iteration = 0  # starting count
    best_val_loss = 1000
    eval_steps = int((count - train_stop) // bs) + 2
    eval_every_nsteps = min(50, steps_per_ep)
    best_val_index = 0

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        triplets, labels = next(train_generator)
        loss = model.train_on_batch(triplets, labels)
        n_iteration += 1
        if i % eval_every_nsteps == 0:
            print("{}/{} -------------".format(i, n_iter))
            print(
                "[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time() - t_start) / 60.0,
                                                                                      loss, n_iteration))
            val_loss = []
            for ii in range(eval_steps):
                data, labels1 = next(test_generator)
                val_loss.append(model.predict_on_batch(data))
            curr_val_loss = np.mean(np.mean(val_loss))
            print("val_loss = ", curr_val_loss)
            if (curr_val_loss < best_val_loss):
                print("best loss found, previous: {}, current: {} ".format(best_val_loss, curr_val_loss))
                best_val_loss = curr_val_loss
                best_val_index = i
                print("curr best_val_index= ", best_val_index)
                base_model.save_weights(modelFilePath + "dev_base_Res50_best.hdf5")
                # model.save_weights(modelFilePath + "complete_res50_best.hdf5")
        if ((n_iteration - best_val_index) > patience * steps_per_ep):
            print("best val loss={}, at iter={}".format(best_val_loss, best_val_index))
            break
    model.save_weights(modelFilePath + "complete_final.hdf5")
    base_model.save_weights(modelFilePath + "base_Res50.hdf5")
    findAccuracy(base_model)
    return model, base_model


def main():
    # Get command line arguments
    print("inside main")
    '''args= getArgOptions()
    q_image_list,q_ids = getImIds(args.anchor_file_list,args.anchor_img_dir)
    ref_image_list, ref_ids = getImIds(args.ref_file_list, args.ref_img_dir)
    global Q_List, R_List
    global Q_IDS, REF_IDS
    Q_List = q_image_list
    R_List = ref_image_list
    Q_IDS = q_ids
    REF_IDS = ref_ids
    
    import shutil
    for f in q_image_list:
        shutil.copy(f, 'D:\\prjs\\im-similarity\\data\\subset\\query')
    exit()
    '''

    #generate_subset_embeddings()
    #save_QueryImagesAsHdf5()
    #save_RefImagesAsHdf5()
    #generate_full_QueryEmbeddings()
    #trainLoop()
    #analyze_subsetAcc()

    #generate_full_RefEmbeddings()

    #save_subsetImagesAsHdf5()
    #generate_subset_embeddings()

    trainLoop()
    #generate_full_QueryEmbeddings('models/weights/OFF_Embeddings_res50_best.hdf5','off')
    #generate_full_QueryEmbeddings('models/weights/SMHD_Embeddings_best.hdf5',"_SMHD")
    #generate_full_QueryEmbeddings('models/weights/resnet50Reg0.8Embeddings_final.hdf5','_2day')
    #generate_full_QueryEmbeddings('models/weights/resnet50Regbase_90_.hdf5','_colab90')

    #generate_full_RefEmbeddings()
    #train_public_gt(epochs=20, bs=32)
    #mytest_hdf5loader()

    #name="resnet50good"
    #generate_full_embeddings()



if __name__ == '__main__':
    main()