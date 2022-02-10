import numpy as np
import cv2
import matplotlib.pyplot as plt
from settings import * # importing all the variables and Cosntants
#from models.resnet50good import preprocess

def plot_triplets(examples):
    plt.figure(figsize=(6, 2))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.squeeze(examples[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def read_image(imname):
    """Choose an image from our training or test data with the
    given label."""
    im = cv2.imread(imname)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img= img/255
    img = cv2.resize(img,IM_SIZE )

    return img


def get_triplet(idx):
    """Choose a triplet (anchor, positive, negative) of images
    such that anchor and positive have the same label and
    anchor and negative have different labels."""
    a= read_image(Q_List[idx])
    p = read_image(R_List[idx])
    #generate random index for neg
    negidx = min(4991, idx+ np.random.randint(0,4991-idx))
    n = read_image(R_List[negidx])

    return a, p, n

def generate_triplets(start=0,stop=4991, BATCH_SIZE=BATCH_SIZE):
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

        A = np.array(list_a, dtype='float32')
        P = np.array(list_p, dtype='float32')
        N = np.array(list_n, dtype='float32')
        #A = preprocess(np.array(list_a, dtype='float32'))
        #P = preprocess(np.array(list_p, dtype='float32'))
        #N = preprocess(np.array(list_n, dtype='float32'))
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
    A = np.array(list_imgs, dtype='float32')
    #A = preprocess(np.array(list_imgs, dtype='float32'))
    # a "dummy" label which w
    return A,list_ids

def compute_dist(a,b):
    return np.sum(np.square(a-b))
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
    # Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size), selection), norm_batchs_size, replace=False)

    selection = np.append(selection, selection2)

    triplets = [studybatch[0][selection, :, :, :], studybatch[1][selection, :, :, :], studybatch[2][selection, :, :, :]]
    label = np.ones(len(selection))

    return triplets,label