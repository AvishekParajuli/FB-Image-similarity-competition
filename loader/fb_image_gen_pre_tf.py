import numpy as np
import cv2
import matplotlib.pyplot as plt
from settings import * # importing all the variables and Cosntants
from getmodel import *
#from models.resnet50tf import preprocess, get_model_name
from augument import * # add_noise
import time


def myModelName():
    return get_model_name()

def plot_triplets(examples,subtitle=["query", "predicted", "GD_Truth"]):
    plt.figure(figsize=(18,18))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.squeeze(examples[i]))
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.title(subtitle[i])
    plt.show()

def plot_pairs(examples,subtitle=["query","pred"], title =""):
    #plt.figure(figsize=(6, 2))
    for i in range(2):
        plt.subplot(1, 2, 1 + i)
        plt.imshow(np.squeeze(examples[i]))
        plt.xticks([])
        plt.yticks([])
        plt.title(subtitle[i])
    plt.title(title)
    plt.show()
def squeezeifrequired(img):
    if img.ndim==4:
        return np.squeeze(img)
    else:
        return img

def plot_triplets_batch(batchdata, numTriplets =1, start=0):
    labels = batchdata[1]#contain ones

    q = batchdata[0][0]
    a = batchdata[0][1]
    n = batchdata[0][2]
    batchSize = q.shape[0]
    plt.figure(figsize=(6, 2))
    if batchSize ==1:
        for i in range(numTriplets):
            plt.subplot(i + 1, 3, 1 + 3 * i)
            plt.imshow(np.squeeze(q))#q[i] is also ok
            plt.xticks([])
            plt.yticks([])
            plt.title("query")
            plt.subplot(i + 1, 3, 2 + 3 * i)
            plt.imshow(np.squeeze(a))
            plt.title("anchor")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(i + 1, 3, 3 + 3 * i)
            plt.imshow(np.squeeze(n))
            plt.title("negative")
            plt.xticks([])
            plt.yticks([])
    else:
        for i in range(numTriplets ):
            plt.subplot(numTriplets, 3, 1 +3*i)
            plt.imshow(np.squeeze(q[i+start]))
            plt.xticks([])
            plt.yticks([])
            plt.title("query")
            plt.subplot(numTriplets, 3, 2+3*i)
            plt.imshow(np.squeeze(a[i+start]))
            plt.title("anchor")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(numTriplets, 3, 3+3*i)
            plt.imshow(np.squeeze(n[i+start]))
            plt.title("negative")
            plt.xticks([])
            plt.yticks([])
    plt.show()

def plot_batches(batchdata):
    images = batchdata[1]
    names = batchdata[0]
    ntotal = len(images)
    plt.figure(figsize=(6, 2))
    for i in range(ntotal):
        plt.subplot(1, ntotal, 1 + i)
        plt.imshow(np.squeeze(images[i]))
        plt.title(names[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def sanitizeNegativeData(selected_q_ids, ipnegdata):
    from copy import deepcopy
    negdata = deepcopy(ipnegdata)#use deepcopy for tuple
    totallen = len(negdata[0])
    for indx,value  in enumerate(negdata[0]):
        matchingIdxs=[]
        if (value == selected_q_ids[indx]):
            #print("same ids found at {}; swaping with {} ".format(indx,(indx-1)%totallen))
            currData = negdata[1][indx].copy()
            negdata[0][indx] = negdata[0][(indx-1)%totallen] #chanigng the ids
            negdata[1][indx] = negdata[1][(indx - 1) % totallen].copy() # chanigng the image data

            negdata[0][(indx - 1) % totallen] = value
            negdata[1][(indx - 1) % totallen] = currData

    return negdata




def read_image(imname):
    """Choose an image from our training or test data with the
    given label."""
    im = cv2.imread(imname)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #img= img/255
    img = cv2.resize(img,IM_SIZE, interpolation=cv2.INTER_AREA)

    return img


def get_triplet(idx, negidx=-1, mode =''):
    """Choose a triplet (anchor, positive, negative) of images
    such that anchor and positive have the same label and
    anchor and negative have different labels."""
    a= read_image(Q_List[idx])
    if mode =='train':
        a = apply_augumentation(a)
    p = read_image(R_List[idx])
    if negidx== -1:
      #generate random index for neg
      negidx = min(4991, idx+ np.random.randint(0,4991-idx))
    n = read_image(R_List[negidx])

    return a, p, n

def generate_triplets(start=0,stop=4991, BATCH_SIZE=BATCH_SIZE, mode =''):
    """Generate an un-ending stream (ie a generator) of triplets for
    training or test."""
    np.random.RandomState(seed=32)
    while True:
        list_a = []
        list_p = []
        list_n = []

        for i in range(BATCH_SIZE):
            #get random idx
            idx = np.random.randint(start, stop)
            a, p, n = get_triplet(idx)
            list_a.append(a)#anchor(query)
            list_p.append(p)#positive(reference)
            list_n.append(n)#negative(reference)

        A = preprocess(np.array(list_a, dtype='float32'))
        P = preprocess(np.array(list_p, dtype='float32'))
        N = preprocess(np.array(list_n, dtype='float32'))
        #A = np.array(list_a, dtype='float32')
        #P = np.array(list_p, dtype='float32')
        #N = np.array(list_n, dtype='float32')
        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(BATCH_SIZE)
        yield [A, P, N], label

def generate_triplets_train_hdfseq(start=0, stop=50_000, batch_sz=100, forcePrep = True):
    """Generate an un-ending stream (ie a generator) of triplets for
    training images.
    forcePrep= False is used for testing; else it shouldbe default True
    """


    train_hdf5_file = './data/image/image_train_0_chunk100.hdf5'
    neg_ref_file = './data/image/image_extended_Ref.hdf5'#fastest due to contiguous +chunks=100

    # since this uses a single hdf5 file, it always needs to augument
    # THerefore ref_loader always has to do prep= False; it will call preprocess later if forcePrep=True(default)
    ref_loader = Hdf5Sequence(train_hdf5_file, idlist='', batch_size=batch_sz,prep=False)# this is chunk of 100
    neg_loader = Hdf5Sequence(neg_ref_file, idlist='', batch_size=batch_sz,prep=forcePrep)

    datalenRef = len(ref_loader)
    print("Inside: generate_triplets_hdfseq: total seq data= {}".format(datalenRef))
    currentRow =0
    np.random.RandomState(seed=32)

    while True:

        list_a = np.zeros((batch_sz,160,160,3),dtype='uint8')
        list_p = np.zeros((batch_sz,160,160,3),dtype='uint8')
        list_n = np.zeros((batch_sz,160,160,3),dtype='uint8')
        #print("currentRow = ",currentRow)
        timeStart = time.time()
        selected_q_ids =[]
        curridx = np.random.randint(start, stop)
        posdata = ref_loader[curridx]

        for i in range(batch_sz):
            p = posdata[1][i]
            curr_q_id = posdata[0][i]
            selected_q_ids.append(curr_q_id)
            try:
                a = apply_augumentation(p)
                list_a[i, :, :, :] = np.squeeze(a)
            except Exception as e:
                print("error at idx={}".format(i))
                #print(e)
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
                list_a[i, :, :, :] = np.squeeze(p)


        currentRow += 1

        list_p = np.squeeze(posdata[1])


        negidx = np.random.randint(0, datalenRef)
        # print("current negidx= ",negidx)
        negdata = ref_loader[negidx]#[currentRow]
        #negdata = posdata
        outnegData = sanitizeNegativeData(selected_q_ids, negdata)
        list_n = np.squeeze(outnegData[1])
        if forcePrep:
            # now perform preprocessing for all anchor(augumented), and pos
            list_a = preprocess(np.array(list_a, dtype='float32'))  # since query image wasn't preprocessed
            list_p = preprocess(np.array(list_p, dtype='float32'))
            list_n = preprocess(np.array(list_n, dtype='float32'))

        label = np.ones((batch_sz,1))
        timestop = time.time()
        if DEBUG:
            print("Time for batchdata collection {} mins".format((timestop - timeStart) / 60))
        yield [list_a, list_p, list_n], label


def generate_dev_triplets(public_gt_array,batch_sz=16):

    subset_ref_filename = './data/image/im_subset_ref.hdf5'
    dev_query_filename = './data/image/image_dev_queries.hdf5'
    neg_ref_filename = './data/image/image_extended_Ref.hdf5'

    ref_loader = Hdf5Sequence(subset_ref_filename, idlist='', batch_size=1)
    query_loader = Hdf5Sequence(dev_query_filename, idlist='', batch_size=1)
    neg_loader = Hdf5Sequence(neg_ref_filename, idlist='', batch_size=1)
    TOTAL = 25000  # 25k

    ref_counter = 0
    neg_counter = 0
    image = np.zeros((160, 160, 3))
    pos = np.zeros((160, 160, 3))
    neg = np.zeros((160, 160, 3))
    # batchData = np.zeros(shape =(bs,160,160,3))
    list_a = []
    list_p = []
    list_n = []
    lenArr = len(public_gt_array)
    for idx, row in enumerate(public_gt_array):
        q_id, r_id = row
        q_d = query_loader[idx]
        # q_d[0]= id, q_d[1] == image
        # row[0] = q_id= q_d[idx] always
        image = q_d[1]
        if r_id != '':
            r_d = ref_loader[ref_counter]
            ref_counter += 1
            if row[1] == r_d[0]:
                pos = r_d[1]
            else:
                pos = apply_augumentation(image)
        else:
            pos = apply_augumentation(image)
            # add noisy image of query as positive refernece
        n_d = neg_loader[neg_counter]
        if row[0] == n_d[0]:
            neg = n_d[1]
            neg_counter += 1
        else:
            neg = ref_loader[idx + np.random.randint(1, 100)][1]
        list_a.append(image)
        list_p.append(pos)
        list_n.append(neg)
        if idx%batch_sz ==0:
            list_a = []
            list_p = []
            list_n = []
            yield [np.array(list_a), np.array(list_p), np.array(list_n)], np.ones(batch_sz)

def generate_triplets_hdfseq(start=0,stop=4991, batch_sz=16,forcePrep=True):

    subset_ref_filename = './data/image/im_subset_ref.hdf5'
    query_filename = './data/image/im_subset_query.hdf5'
    #neg_ref_filename =  './data/image/mergedRefExtended_0to2.hdf5'
    neg_ref_filename = './data/image/image_extended_Ref.hdf5'# this is faster due to smaller size or contiguous
    #neg_ref_filename = './data/image/mergedRefExtended0_2_chunk100.hdf5'  # this is faster due to chunksize=100
    #neg_ref_filename = './data/image/mergedRefExtended0_2_chunk100_cont.hdf5'#fastest due to contiguous +chunks=100

    #this will never have augumentation

    query_loader = Hdf5Sequence(query_filename, idlist='', batch_size=1,prep=forcePrep)
    ref_loader = Hdf5Sequence(subset_ref_filename, idlist='', batch_size=1,prep =forcePrep)
    neg_loader = Hdf5Sequence(neg_ref_filename, idlist='', batch_size=batch_sz,prep=forcePrep)
    datalenNeg = len(neg_loader)
    datalenRef = len(ref_loader)
    print("Inside: generate_triplets_hdfseq: total seq data= {}, neg data= {}".format(datalenRef, datalenNeg*batch_sz))
    currentRow =0

    while True:
        list_a = np.zeros((batch_sz,160,160,3),dtype='uint8')
        list_p = np.zeros((batch_sz,160,160,3),dtype='uint8')
        list_n = np.zeros((batch_sz,160,160,3),dtype='uint8')
        #print("currentRow = ",currentRow)

        timeStart = time.time()
        selected_q_ids =[]

        for i in range(batch_sz):
            # get random idx
            idx = np.random.randint(start, stop)
            a= query_loader[idx][1]
            p = ref_loader[idx][1]
            curr_q_id = ref_loader[idx][0]
            selected_q_ids.append(curr_q_id)
            #negidx = np.random.randint(start, stop)
            #print("curr_q_id = {}, idx={}, negidx={}".format(curr_q_id,idx,negidx))
            '''
            while neg_loader[negidx][0]==curr_q_id:
                #worst case scenario when negID equal the anchor ID
                print("inside here, ")
                negidx = np.random.randint(0, datalenNeg)
                # try until the ids are different
            '''
            #n = ref_loader[negidx][1]

            #a, p, n = get_triplet(idx, mode=mode)
            #list_a.append(np.squeeze(a))  # anchor(query)
            #list_p.append(np.squeeze(p))  # positive(reference)
            #list_n.append(np.squeeze(n))  # negative(reference)
            #print(" i={}, negidx ={} ".format(i, negidx))
            try:
                list_a[i, :, :, :] = np.squeeze(a)
                list_p[i, :, :, :] = np.squeeze(p)
                #list_n[i, :, :, :] = np.squeeze(n)
            except :
                print("error at idx={}, and negidx={}".format(idx,negidx))
        negidx = np.random.randint(0, datalenNeg)
        #print("current negidx= ",negidx)
        negdata = neg_loader[negidx]#[currentRow]
        outnegData = sanitizeNegativeData(selected_q_ids, negdata)
        list_n= np.squeeze(outnegData[1])

        currentRow += 1
        label = np.ones((batch_sz,1))
        timestop = time.time()
        if DEBUG:
            print("Time for batchdata collection {} mins".format((timestop - timeStart) / 60))
        #yield [np.array(list_a), np.array(list_p), np.array(list_n)], label
        yield [list_a, list_p, list_n], label


def apply_augumentation(image):
    img1 = apply_augumentaion_wrapper(image)
    #img1 = add_noise(image)
    #img1 = image
    return img1

def generate_offline_triplets(qryIds,negIds,start, stop, BATCH_SIZE=BATCH_SIZE, mode=''):
    """Generate an un-ending stream (ie a generator) of triplets for
    training or test."""
    #np.random.RandomState(seed=32)
    while True:
        list_a = []
        list_p = []
        list_n = []
        stop = len(qryIds)

        for i in range(BATCH_SIZE):

            if i%2==0:
                # get offline hard index:
                idx = qryIds[np.random.randint(start, stop)]
                #idx=3
                #print("**idx = ", idx)
                #now find respective negId pertaining to idx
                myindex= qryIds.index(idx)
                #print("***myindex == ",myindex)
                negId = negIds[myindex]
                #print("negId = ", negId)
            else:
                negId =-1 # choose random negindex

            a, p, n = get_triplet(idx, negId)
            list_a.append(a)#anchor(query)
            list_p.append(p)#positive(reference)
            list_n.append(n)#negative(reference)

        A = preprocess(np.array(list_a, dtype='float32'))
        P = preprocess(np.array(list_p, dtype='float32'))
        N = preprocess(np.array(list_n, dtype='float32'))
        #A = np.array(list_a, dtype='float32')
        #P = np.array(list_p, dtype='float32')
        #N = np.array(list_n, dtype='float32')
        # a "dummy" label which will come in to our identity loss
        # function below as y_true. We'll ignore it.
        label = np.ones(BATCH_SIZE)
        yield [A, P, N], label


def get_testImages(currIndx,file_List, file_Ids, batch=BATCH_SIZE,):
    """Generate an un-ending stream (ie a generator) of images for test or prediction."""
    list_imgs=[]
    list_ids=[]
    startIdx = currIndx*batch
    for i in range(batch):
        #get random idx
        im = read_image(file_List[startIdx +i])
        list_imgs.append(im)
        list_ids.append(file_Ids[startIdx +i])
    #A = np.array(list_imgs, dtype='float32')
    A = preprocess(np.array(list_imgs, dtype='float32'))
    # a "dummy" label which w
    return A,list_ids

def get_testImages_fast(currIndx,file_List, file_Ids, batch=BATCH_SIZE,):
    """Generate an un-ending stream (ie a generator) of images for test or prediction."""
    list_imgs=[]
    list_ids=[]
    startIdx = currIndx*batch
    for i in range(batch):
        #get random idx
        im = read_image(file_List[startIdx +i])
        list_imgs.append(im)
        list_ids.append(file_Ids[startIdx +i])
    A = np.array(list_imgs, dtype ='uint8')#, dtype='float32')
    #A = preprocess(np.array(list_imgs, dtype='float32'))
    # a "dummy" label which w
    return A,list_ids

def compute_dist(a,b):
    return np.sum(np.square(a-b), axis=1)
def get_batch_hard(network,traingen, draw_batch_size=100,actual_batch_size=32):
    """
       Create batch of APN "hard" triplets

       Arguments:
       draw_batch_size -- integer : number of initial randomly taken samples= 2XBS
       hard_batchs_size -- interger : select the number of hardest samples to keep
       norm_batchs_size -- interger : number of random samples to add
       Returns:
       triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    hard_batchs_size = actual_batch_size//2
    norm_batchs_size = actual_batch_size//2
    # Step 1 : pick a random batch to study
    #studybatch,labels = generate_triplets(start=0,stop=4991, BATCH_SIZE=draw_batch_size)
    studybatch, labels =  next(traingen)

    # Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))

    # Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    # Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A - P), axis=1) - np.sum(np.square(A - N), axis=1)

    # Sort by distance (high distance first) and take the
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    #print("first 5 max loss: ",studybatchloss[0:5])
    # Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size), selection), norm_batchs_size, replace=False)

    selection = np.append(selection, selection2)

    triplets = [studybatch[0][selection, :, :, :], studybatch[1][selection, :, :, :], studybatch[2][selection, :, :, :]]
    label = np.ones(len(selection))

    return triplets,label
def chooseSemiHardNTriplets(ap, an, alpha=ALPHA):
    """
       Create batch of APN "semi-hard" Negativetriplets that statisfies the conditions below:
       cond1: d(A,P)<d(A,N)
       cond2: d(A,N)< d(A,P)+ alpha
    """
    cond1_i = np.where(ap < an)[0]
    cond2_i = np.where(an < ap+alpha)[0]
    #print("cond1_i", cond1_i)
    #print("cond2_i", cond2_i)
    #find intersecting index
    intersection = np.intersect1d(cond1_i, cond2_i)
    #print("\nThe intersection between the two arrays is:\n", intersection)
    #condition: intersection has both cond1, cond2 in common
    outindex = [idx for idx in intersection if ap[idx]<an[idx]]
    #print("outindex length= ", len(outindex))
    return outindex

def get_batch_semihardNeg(network,traingen, draw_batch_size=100,actual_batch_size=32,alpha=ALPHA, hard_perct =0.5):
    """
       Create batch of APN "semi-hard" Negativetriplets that statisfies the conditions below:
       cond1: d(A,P)<d(A,N)
       cond2: d(A,N)< d(A,P)+ alpha
       
       Arguments:
       draw_batch_size -- integer : number of initial randomly taken samples= 2XBS
       hard_batchs_size -- interger : select the number of hardest samples to keep
       norm_batchs_size -- interger : number of random samples to add
       Returns:
       triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    hard_batchs_size = int(hard_perct*actual_batch_size)
    norm_batchs_size = actual_batch_size-hard_batchs_size
    # Step 1 : pick a random batch to study
    #studybatch,labels = generate_triplets(start=0,stop=4991, BATCH_SIZE=draw_batch_size)
    studybatch, labels =  next(traingen)

    # Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))

    # Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    # Compute d(A,P)and d(A,N)
    #studybatchloss = np.sum(np.square(A - P), axis=1) - np.sum(np.square(A - N), axis=1)
    dAP = np.sum(np.square(A - P), axis=1)
    dAN = np.sum(np.square(A - N), axis=1)
    semihardNIdx = chooseSemiHardNTriplets(dAP,dAN, alpha)

    # Sort by distance (high distance first) and take the
    #selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]#selection gives us index;
    allowedLen = min(hard_batchs_size, len(semihardNIdx))
    selection = semihardNIdx[:allowedLen]
    
    if(len(semihardNIdx)==0) or (len(semihardNIdx)< 0.5*hard_batchs_size):
        #print("no semi hard found, going for the maximum loss: ")
        studybatchloss = dAP - dAN
        selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    elif allowedLen<hard_batchs_size:
        norm_batchs_size += hard_batchs_size-allowedLen
    # Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size), selection), norm_batchs_size, replace=False)

    selection = np.append(selection, selection2)

    triplets = [studybatch[0][selection, :, :, :], studybatch[1][selection, :, :, :], studybatch[2][selection, :, :, :]]
    label = np.ones(len(selection))

    return triplets,label


import tensorflow
import h5py
import math
class Hdf5Sequence(tensorflow.keras.utils.Sequence):

    def  __init__(self, myHd5File,idlist, batch_size, offset=0,prep=True):
        self.srcFile,self.Ids =  myHd5File,idlist
        self.batch_size = batch_size
        self.hh = h5py.File(myHd5File, "r")#call these only once
        self.data_size = self.hh["vectors"].shape[0]#
        self.size_per_batch = self.data_size/batch_size
        self.img_size = (160,160)#always for this case/project
        self.offset = offset
        self.prep = prep

    def __len__(self):
        return math.floor(self.data_size / self.batch_size)
    def my_preprocess_img(self, imgBatch):
        return preprocess(np.array(imgBatch, dtype='float32'))

    def __getitem__(self, idx):
        """Return tuple(input, id) or (img, id) correspondidng to batch #idx
                single Call to getitem will return batch_size length of data"""
        idx = idx % self.__len__()
        # start = current * size_per_batch
        # end = (current + 1) * size_per_batch
        startIndex = self.offset + idx * self.batch_size
        stopIndex = self.offset+ startIndex + self.batch_size
        descs = []
        names = []
        imgdata_batch = self.hh["vectors"][startIndex:stopIndex]
        if self.prep:
            imgdata_batch = self.my_preprocess_img(imgdata_batch)
            #else do not preprocess
        descs.append(imgdata_batch)
        names += np.array(self.hh["image_names"][startIndex:stopIndex], dtype=object).astype(str).tolist()
        '''

        batch_ip_img_paths = self.input_img_paths[startIndex: stopIndex]
        batch_ip_mask_paths = self.input_mask_paths[startIndex: stopIndex]
        batch_imgs = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        


        # both input_img and target_img will have size of img_size=(160,160)
        # x shape =(32,H,W,3) NHWC format  i.e. 4D tensor
        for ii in range(self.batch_size):
            img = read_image(batch_ip_img_paths[ii])
            mask = read_mask(batch_ip_mask_paths[ii])
            if self.aug != None:
                img, mask = augment_image_and_mask(img, mask, self.aug)
            img = normalize_img(img)
            batch_imgs[ii] = img
            batch_masks[ii] = mask
        return batch_imgs
        '''
        return names, np.vstack(descs)

