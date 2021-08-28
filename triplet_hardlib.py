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

#from models.resnet_50good import *
from models.resnet50Reg import *
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard
from importlib import reload
import loader
reload (loader)
#from loader.fb_image_gen import *
from loader.fb_image_gen_pre import *
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
def train_basic(model, base_model, epochs=10):
    image_count= len(Q_List)
    train_stop_idx = 0.9*image_count
    bs= 64 #32

    train_generator = generate_triplets(start=0,stop=train_stop_idx,BATCH_SIZE=bs)
    test_generator = generate_triplets(start=train_stop_idx+1, stop=image_count-1, BATCH_SIZE=bs)
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
    model_save_name = modelFilePath+"model_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    embeddings_save_name = modelFilePath+"em_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    print("model weights filepath name is: ",model_save_name)

    stps= train_stop_idx//bs
    my_callbacks =[EarlyStopping(patience=10),
                   ModelCheckpoint(filepath=modelFilePath+'model.h5',
                                   save_weights_only=True,save_best_only=True,monitor='val_loss'),
                   TensorBoard(log_dir='./models/logs')]
    valid_Stps = (image_count-train_stop_idx)//bs
    history = model.fit_generator(train_generator,epochs=EPOCHS, steps_per_epoch=stps,
                                  validation_data=test_generator,
                                  validation_steps=valid_Stps,
                                  callbacks=my_callbacks, verbose=1)
    
    testgen = generate_triplets(BATCH_SIZE=1)
    adata = next(testgen )

    pred = model.predict(adata[0])
    a_em = np.squeeze(base_model.predict(adata[0][0]))
    b_em = np.squeeze(base_model.predict(adata[0][1]))
    c_em = np.squeeze(base_model.predict(adata[0][2]))

    print("difference between anchor and Positive:",np.sum(a_em-b_em))
    print("difference between anchor and negative:", np.sum(a_em - c_em))

    '''
    model.save_weights(model_save_name)
    base_model.save_weights(embeddings_save_name)


    testgen = generate_triplets(BATCH_SIZE=1)
    adata = next(testgen )

    pred = model.predict(adata[0])
    a_em = np.squeeze(base_model.predict(adata[0][0]))
    b_em = np.squeeze(base_model.predict(adata[0][1]))
    c_em = np.squeeze(base_model.predict(adata[0][2]))

    print("difference between anchor and Positive:",np.sum(a_em-b_em))
    print("difference between anchor and negative:", np.sum(a_em - c_em))
    #this sum(square) metric is better
    print("sum(square)difference between anchor and Positive:", np.sum(np.square(a_em - b_em)))
    print("sum(square)difference between anchor and negative:", np.sum(np.square(a_em - c_em)))


    import time
    timestart = time.time()
    c1_em = base_model.predict(data[0][2])#directly predict on batch to speed up
    timestop = time.time()
    print("Time for prediction {} ms".format((timestop - timestart)*1000))

    timestart = time.time()
    score = cosine_similarity(a_em, b_em)
    timestop = time.time()
    print("Time for cosine similarity{} ms".format((timestop - timestart) * 1000))
    print("cosine sim between anchor and Positive:", score)
    print("cosine similarity between anchor and negative:", cosine_similarity(a_em, c_em))
    '''
    return model, base_model


def test_hardbatch(model, base_model, epochs):
    image_count = len(Q_List)
    train_stop_idx = int(0.8 * image_count)

    largeBS = 60
    bs=32
    large_Generator = generate_triplets(start=0, stop=train_stop_idx, BATCH_SIZE=largeBS)
    #base_model = embedding_model()
    #triplets, labels = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)# if return
    #hardbatch_gen = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)

    test_generator = generate_triplets(start=train_stop_idx+1, stop=image_count-1, BATCH_SIZE=bs)
    #model = complete_model(base_model)
    #model.compile(loss=identity_loss, optimizer=Adam(1e-4))
    import time
    from datetime import datetime

    modelFilePath = "./models/weights/"
    #base_model.load_weights(modelFilePath + "Embeddings_best.hdf5")
    #model.load_weights(modelFilePath + "complete_res18_best.hdf5")
    EPOCHS = epochs


    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_H%H_M%M")
    model_save_name = modelFilePath + "model_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    embeddings_save_name = modelFilePath + "em_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    print("model weights filepath name is: ", model_save_name)

    #history = model.fit_generator(hardbatch_gen, epochs=2, steps_per_epoch=10,
    #                              validation_data=test_generator, validation_steps=10)

    steps_per_ep = int(train_stop_idx//bs)
    steps_per_eval = int((image_count-train_stop_idx)//bs)
    n_iter = steps_per_ep*EPOCHS
    n_iteration=0#starting count
    best_val_loss = 1000
    eval_every = min(100,steps_per_ep)

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        #triplets,labels = get_batch_hard(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs)
        triplets,labels = get_batch_semihardNeg(base_model, large_Generator, draw_batch_size=largeBS,actual_batch_size=bs, alpha=0.8)
        loss = model.train_on_batch(triplets, labels)
        n_iteration += 1
        if i % eval_every == 0:
            print("{}/{} -------------".format(i,n_iter))
            print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time() - t_start) / 60.0,
                                                                                      loss, n_iteration))
            val_loss = []
            for i in range(steps_per_eval):
                data, labels = next(test_generator)
                val_loss.append(model.predict_on_batch(data))
            curr_val_loss = np.mean(np.mean(val_loss))
            print("val_loss = ", curr_val_loss)
            if(curr_val_loss <best_val_loss):
                print("best loss found, previous: {}, current: {} ".format(best_val_loss,curr_val_loss))
                best_val_loss = curr_val_loss
                base_model.save_weights(modelFilePath + "SMHDN_Em_best.hdf5")
                #model.save_weights(modelFilePath + "complete_res18_best.hdf5")

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

def test_hardOfflineBatch(model, base_model, epochs):
    Isaved = pickle.load(open("./L2Index_2_prev.p", "rb"))
    queryId, negId = getHardNegList(Isaved)

    count = len(queryId)
    train_stop = int(0.8 * count)
    print("train_stop= ", train_stop)
    bs = 64 #32
    EPOCHS = epochs #10
    patience = 10  # in epochs
    train_generator = generate_offline_triplets(queryId, negId, 0, train_stop, BATCH_SIZE=bs)
    test_generator = generate_offline_triplets(queryId, negId, train_stop + 1, count, BATCH_SIZE=bs)


    #model = complete_model(base_model)
    #model.compile(loss=identity_loss, optimizer=Adam(1e-4))

    modelFilePath = "./models/weights/"
    #base_model.load_weights(modelFilePath + "Embeddings_best.hdf5")
    #model.load_weights(modelFilePath + "complete_res18_best.hdf5")
    '''
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_H%H_M%M")
    model_save_name = modelFilePath + "model_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    embeddings_save_name = modelFilePath + "em_" + get_model_name() + "_EP" + str(EPOCHS) + "_" + dt_string + ".hdf5"
    print("model weights filepath name is: ", model_save_name)
    '''

    #history = model.fit_generator(hardbatch_gen, epochs=2, steps_per_epoch=10,
    #                              validation_data=test_generator, validation_steps=10)

    steps_per_ep = int(train_stop//bs)
    n_iter = steps_per_ep*EPOCHS
    n_iteration=0#starting count
    best_val_loss = 1000
    eval_steps = int((count - train_stop)//bs)+2
    eval_every_nsteps = min(50, steps_per_ep)
    best_val_index = 0

    print("Starting training process!")
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
                base_model.save_weights(modelFilePath + "OFF_Em_res50_best.hdf5")
                #model.save_weights(modelFilePath + "complete_res50_best.hdf5")
        if ((n_iteration - best_val_index) > patience * steps_per_ep):
            print("best val loss={}, at iter={}".format(best_val_loss, best_val_index))
            break
    #model.save_weights(modelFilePath + "complete_final.hdf5")
    #base_model.save_weights(modelFilePath + "Embeddings_final.hdf5")
    findAccuracy(base_model)
    return model, base_model

def trainLoop():
    base_model = embedding_model()
    model = complete_model(base_model)
    model.summary()
    model.compile(loss=identity_loss, optimizer=Adam(1e-4))
    modelFilePath = "./models/weights/"
    #model, base_model = train_basic(model, base_model, epochs=20)
    #model, base_model = test_hardOfflineBatch(model, base_model, epochs=10)
    model.load_weights(modelFilePath+"complete_res50_best.hdf5")
    base_model.load_weights(modelFilePath + "Embeddings_res50_best.hdf5")
    findAccuracy(base_model)
    model, base_model = test_hardbatch(model, base_model, epochs=10)
    findAccuracy(base_model)
    model.save_weights(modelFilePath +get_model_name()+ "complete_final.hdf5")
    base_model.save_weights(modelFilePath + get_model_name()+"Embeddings_final.hdf5")

def generate_subset_embeddings():
    base_model = embedding_model()
    base_model.load_weights("./models/weights/" + "Embeddings_res50_best.hdf5")
    findAccuracy(base_model, save=True)

def generate_full_embeddings():
    base_model = embedding_model()
    base_model.load_weights("./models/weights/" + "Embeddings_res50_best.hdf5")
    name = get_model_name()
    #anchor_file_list = './list_files/subset_1_queries'
    #anchor_img_dir = "D:/prjs/im-similarity/data/query"
    #q_image_list, q_ids = getImIds('./list_files/dev_queries', anchor_img_dir)
    #XQ = gen_embeddings(base_model,file_list=q_image_list, file_ids=q_ids, outFile='./data/full_query_em_' + name + '.hdf5')
    ref_file_list = './list_files/references'
    ref_img_dir = 'C:/Users/parajav/PycharmProjects/isc/reference/reference'
    ref_image_list, ref_ids = getImIds(ref_file_list, ref_img_dir)
    print("totoal IDS:", len(ref_ids))
    interval = 100000#100K
    iters = int(len(ref_ids)/interval)
    print("iters", iters)
    for ii in range(1,iters):
        i0 =ii*interval
        i1 = (ii+1)*interval
        print("reading files from {} to {}".format(i0, i1))
        XR = gen_embeddings(base_model,file_list=ref_image_list[i0:i1], file_ids=ref_ids[i0:i1],
                        outFile='./data/full_ref_em_' +str(ii)+ name + '.hdf5')
    #findAccuracy(base_model, save=True)

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
    #image_count = len(q_image_list)
    #gen_embeddings(args)
    #trainLoop()
    #name="resnet50good"
    generate_full_embeddings()
    #anchor_img_dir = "D:/prjs/im-similarity/data/query"
    #q_image_list, q_ids = getImIds('./list_files/dev_queries', anchor_img_dir)
    #gen_imageEmbeddings(q_image_list, q_ids, outFile='./imageee_querydev_small.hdf5')
    #print(len(q_image_list))
    #from isc.io import write_hdf5_descriptors, read_descriptors
    #query_image_ids, xq = read_descriptors(['./dummy.hdf5'])
    #print(len(query_image_ids))


    #gen_embeddings( file_list=Q_List, file_ids=Q_IDS, outFile='./data/subset_query_em_'+name+'.hdf5')
    #gen_embeddings(file_list=R_List, file_ids=REF_IDS, outFile='./data/subset_ref_em_'+name+'.hdf5')
    #analyzeEmbedingMetrics(['./data/subset_query_em_'+name+'.hdf5'],
    # ['./data/subset_ref_em_'+name+'.hdf5'])
    #analyzeEmbedingMetrics(['./data/subset_query_embedings_resnet18_best.hdf5'],
    #                      ['./data/subset_ref_embedings_resnet18_best.hdf5'])
    #test_generator()
    #test_hardbatch()
    #test_savedmodels()


if __name__ == '__main__':
    main()