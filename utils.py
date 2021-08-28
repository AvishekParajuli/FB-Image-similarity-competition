from loader.fb_image_gen_pre import *
import pickle
import datetime
import time
import faiss
from settings import * # importing all the variables and Cosntants

def norm_vec(x):
    output = np.sqrt(max(sum(x**2), 1e-12))
    return output
def cosine_similarity(a,b):
    out = sum(np.multiply(a,b)/(norm_vec(a)*norm_vec(b) ))
    return out
def getMatchingScore(I , k=1):
    lenI = len(I)
    if k==1:
        matches = [index for index, value in enumerate(I) if value[0]== index]
    else:
        matches = [index for index, value in enumerate(I) if index in value[0:k]]
    non_matching = [i for i in range(lenI) if i not in matches]
    print("Top K={} matchign accuracy {}% : ".format(k, len(matches)/lenI *100))

def gen_embeddings(base_model, file_list, file_ids, outFile):
    from tqdm import tqdm
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('')))

    #base_model = embedding_model()
    #base_model.load_weights('./embeddings_baseline_triplet_im160_ep50.hdf5')
    #base_model.load_weights('./models/weights/em_resnet50_EP14_28-07-2021_H17_M52.hdf5')
    #base_model.load_weights('./models/weights/em_resnet18_EP50_14-08-2021_H12_M59.hdf5')
    #base_model.load_weights('./models/weights/em_resnet50good_EP30_15-08-2021_H01_M03.hdf5')
    #print("Loaded model weights")
    batch =32
    steps = len(file_list)//batch
    id_Arr=[]
    embedding_Arr=np.array((batch,256))
    print("Inference for generating embedding. THis will take long....")
    timestart = time.time()
    for idx in tqdm(range(steps)):
        data=  get_testImages( batch=batch,file_List=file_list, file_Ids=file_ids,currIndx= idx)
        pred = base_model.predict(data[0])
        pred_array = np.squeeze(pred)#(batch,64)
        if idx==0:
            embedding_Arr=pred_array
        else:
            embedding_Arr=np.vstack((embedding_Arr,pred_array))
        id_Arr.append(data[1])# not used
    timestop = time.time()
    print("Time for generation {} mins".format((timestop - timestart)/60))
    if outFile != '':
        from isc.io import write_hdf5_descriptors, read_descriptors
        print("Writing embedding to file{}".format(outFile))
        write_hdf5_descriptors(embedding_Arr, file_ids[0:len(embedding_Arr)], outFile)
    else:
        print("skipping writing embeddings to file")
        # query_image_ids, xq = read_descriptors(['../data/ref_embedings_1.hdf5'])
    return embedding_Arr
def gen_imageEmbeddings( file_list, file_ids, outFile):
    from tqdm import tqdm
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('')))
    batch =32
    steps = len(file_list)//batch
    id_Arr=[]
    embedding_Arr=np.array((batch,256))
    print("Inference for generating embedding. THis will take long....")
    timestart = time.time()
    for idx in tqdm(range(steps)):
        data=  get_testImages_fast( batch=batch,file_List=file_list, file_Ids=file_ids,currIndx= idx)
        array_1d = data[0]#...ravel()
        if idx==0:
            embedding_Arr=array_1d
        else:
            embedding_Arr=np.vstack((embedding_Arr,array_1d))
        #id_Arr.append(data[1])# not used
    timestop = time.time()
    print("Time for generation {} mins".format((timestop - timestart)/60))
    if outFile != '':
        from isc.io import write_hdf5_descriptors, read_descriptors
        print("Writing embedding to file{}".format(outFile))
        write_hdf5_descriptors(embedding_Arr, file_ids[0:len(embedding_Arr)], outFile,type='uint8')
    else:
        print("skipping writing embeddings to file")
        # query_image_ids, xq = read_descriptors(['../data/ref_embedings_1.hdf5'])
    return embedding_Arr

def L2Distance(veca, vecb):
    dist = np.sqrt(np.sum(np.square(veca - vecb)))
    return dist

def L1Distance(veca, vecb):
    dist = np.sum(np.abs(veca - vecb))
    return dist

def getEmbeddingMetrics(query_descs,db_descs):
    from isc.io import  read_descriptors
    import faiss
    db_image_ids, xd = read_descriptors(db_descs)
    query_image_ids, xq = read_descriptors(query_descs)

    print("**********Faisss comparision**************")
    d=256
    k=5
    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    #faiss.normalize_L2(xq)
    index.add(xd)
    D, I = index.search(xq, k)  # sanity check for first 20
    print("**************L2 metric faiss")
    print(I)
    print(D)
    getMatchingScore(I, k=5)
    getMatchingScore(I, 1)
    del index

    index2 = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT) # build the index
    print(index2.is_trained)
    index2.add(xd)
    D2, I2 = index2.search(xq, k)  # sanity check for first 20
    print("**************IP metric faiss")
    print(I2)
    print(D2)
    getMatchingScore(I, k)

    dshape = xd.shape[1]
    num_QImages = len(xq)
    simScores =[]
    L1Scores =[]
    L2Scores =[]
    for ii in range(num_QImages):
        q = xq[ii]
        r = xd[ii]
        simscore= abs(cosine_similarity(q, r))
        l1score = L1Distance(q,r)
        l2score = L2Distance(q,r)
        simScores.append(simscore)
        L1Scores.append(l1score)
        L2Scores.append(l2score)
    return simScores, L1Scores,L2Scores
def analyzeEmbedingMetrics(query_descs,db_descs):
    simScores, L1Scores,L2Scores = getEmbeddingMetrics(query_descs, db_descs)
    '''
    x= range(len(simScores))
    fig, axs = plt.subplots(3)
    fig.suptitle("similarity metrics between each of the matching image IDS")
    axs[0].plot(x, simScores)
    axs[0].set_title(' Cosine similarity')
    axs[1].plot(x, L1Scores)
    axs[1].set_title(' L1 distance')
    axs[2].plot(x, L2Scores)
    axs[2].set_title(' L2 distance')
    print("Cosine similarity: min:{}, max:{}, avg:{}".format(min(simScores), max(simScores), np.mean(simScores)))
    print("L2Scores: min:{}, max:{}, avg:{}".format(min(L2Scores), max(L2Scores), np.mean(L2Scores)))
    simMinId = np.argmin(simScores)
    print("MinCosineSim value ={}, id ={},L2Scores={}".format(simScores[simMinId],
                        simMinId,L2Scores[simMinId]))
    cond = [i for i, value in enumerate(simScores) if value > 0.9]
    condN = [i for i, value in enumerate(simScores) if value < 0.9]
    vec = [L2Scores[i] for i in condN]

    p = plt.hist(simScores, density=True, bins=50)
    plt.ylabel("probability")
    plt.xlabel("data")
    '''

    # see if using combination of cosine similarity and L2scores can help

def findAccuracy(base_model, save= False):
    if save:
        XQ = gen_embeddings(base_model=base_model, file_list=Q_List, file_ids=Q_IDS,
                            outFile='./data/subset_query_'+myModelName()  +'.hdf5')
        XD = gen_embeddings(base_model=base_model, file_list=R_List, file_ids=REF_IDS,
                            outFile='./data/subset_ref_'+myModelName()  +'.hdf5')
    else:
        XQ = gen_embeddings(base_model=base_model, file_list=Q_List, file_ids=Q_IDS, outFile='')
        XD = gen_embeddings(base_model=base_model, file_list=R_List, file_ids=REF_IDS, outFile='')
    # print(XQ.shape)
    # print(XD.shape)

    d = 256
    index = faiss.IndexFlatL2(d)
    k = 1  # k=99 gives 85%
    index.add(XD)
    D, I = index.search(XQ, k)  # search top k
    print("matching index after training....")
    # print(I)
    getMatchingScore(I, k)
    print("I[0:10", I[0:10])
    print("D[0:10", D[0:10])
    return I

def save_images(imname,outdir):
    im = cv2.imread(imname)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IM_SIZE)
    i = np.array(img).astype(np.uint8)
    splitnames = imname.split('/')
    outfilename = outdir+splitnames[-1]
    #print("outfilename:= ", outfilename)
    out = cv2.imwrite(outfilename,i)
    if not out:
        print("Image save failed")

def getMinMax(img, str):
    print("name= {}: min={},max={}, mean={}, type={}".format(str, np.min(img), np.max(img), np.mean(img),img.dtype))

def test_generator():
    train_generator = generate_triplets(start=0, stop=100,BATCH_SIZE=1)
    data = next(train_generator)
    [A, P, N], label = data
    idx=0
    plot_triplets([A[idx],P[idx],N[idx]])
    getMinMax(A[idx], "A")
    getMinMax(P[idx], "P")
    getMinMax(N[idx], "N")

