from loader.fb_image_gen_pre import *
import pickle
import datetime
import time
import faiss
from settings import * # importing all the variables and Cosntants
from tqdm import tqdm
import sys, os
import h5py
from isc.io import write_hdf5_descriptors, read_descriptors


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

def gen_embeddingsSeq(base_model,myHd5File, file_ids, outFile, batch=50):

    batch =batch
    steps = len(file_ids)//batch
    id_Arr=[]
    embedding_Arr=np.array((batch,256))
    hdf5loader = Hdf5Sequence(myHd5File,idlist=file_ids, batch_size=batch)
    print("Inference for generating embedding. THis will take long....")
    timestart = time.time()
    for idx in tqdm(range(steps)):
        data=  hdf5loader[idx]
        pred = base_model.predict(data[1])
        pred_array = np.squeeze(pred)#(batch,64)
        if idx==0:
            embedding_Arr=pred_array
        else:
            embedding_Arr=np.vstack((embedding_Arr,pred_array))
        id_Arr.append(data[0])# not used
    timestop = time.time()
    print("Time for generation {} mins".format((timestop - timestart)/60))
    newids = np.array(id_Arr)
    newids = newids.ravel()
    minlen = min(len(file_ids), len(newids))
    assert (all(newids[0:minlen] == file_ids[0:minlen]))# error if they aren't equal
    if outFile != '':
        from isc.io import write_hdf5_descriptors, read_descriptors
        print("Writing embedding to file{}".format(outFile))
        write_hdf5_descriptors(embedding_Arr, file_ids[0:len(embedding_Arr)], outFile)
        #del embedding_Arr
    else:
        print("skipping writing embeddings to file")
        # query_image_ids, xq = read_descriptors(['../data/ref_embedings_1.hdf5'])
    return embedding_Arr
def gen_imageEmbeddings( file_list, file_ids, outFile, bs):
    from tqdm import tqdm
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('')))
    batch = bs
    steps = len(file_list)//batch
    id_Arr=[]
    #embedding_Arr=np.array((batch,256))
    embedding_Arr = np.zeros((1,1)) #shape does matter just init as numpy array
    print("Saving images as Hdf5. THis will take long....")
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
        print("Writing embedding to file{}".format(outFile))
        write_hdf5_descriptors(embedding_Arr, file_ids[0:len(embedding_Arr)], outFile,type='uint8')
        del embedding_Arr
    else:
        print("skipping writing embeddings to file")
        # query_image_ids, xq = read_descriptors(['../data/ref_embedings_1.hdf5'])

def gen_imageEmbeddingsSlow( file_list, file_ids, outFile, bs):
    #this is memory efficient but 2.7Xslower than simple version
    batch = bs
    steps = len(file_list)//batch
    id_Arr=[]
    #embedding_Arr=np.array((batch,256))
    embedding_Arr = np.zeros((1,1)) #shape does matter just init as numpy array
    print("Saving images as Hdf5. THis will take long....")
    if outFile == '':
        print("skipping writing embeddings to file")
        return
    print("Writing embedding to file{}".format(outFile))
    imgs_count = len(file_ids[0:steps*batch])
    namebyte =  bytes(file_ids[0], "ascii")
    namebytesz= np.array(namebyte).__sizeof__()
    print("total image names: ", len(file_ids[0:steps*batch]))
    timestart = time.time()
    with h5py.File(outFile, 'w') as hf:
        img_data = hf.create_dataset("vectors",shape=(imgs_count,160,160,3), dtype ='uint8'
                                     ,compression="gzip",compression_opts=9)
        #write the ids at the end
        #img_ids = hf.create_dataset("image_names", shape=(total_imgs,namebytesz))
        for idx in tqdm(range(steps)):
            data=  get_testImages_fast( batch=batch,file_List=file_list, file_Ids=file_ids,currIndx= idx)
            #data[0] is image
            vectors = np.ascontiguousarray(data[0], dtype='uint8')
            #image_names =data[1]
            #image_names = np.array([bytes(name, "ascii") for name in image_names ])
            startidx = idx*batch
            stopidx= (idx+1)*batch
            img_data[startidx:stopidx, :, :] = vectors
            #img_ids[idx:(idx + 1) * batch:, :, :] = image_names
        image_names = np.array([bytes(name, "ascii") for name in file_ids[0:imgs_count] ])
        img_ids = hf.create_dataset("image_names", data=image_names)
    timestop = time.time()
    print("Time for generation {} mins".format((timestop - timestart)/60))

def gen_imageEmbeddingsChunks( file_list, file_ids, outFile, bs):
    #this is memory efficient but 2.7Xslower than simple version
    batch = bs
    steps = len(file_list)//batch
    id_Arr=[]
    #embedding_Arr=np.array((batch,256))
    embedding_Arr = np.zeros((1,1)) #shape does matter just init as numpy array
    print("Saving images as Hdf5. THis will take long....")
    if outFile == '':
        print("skipping writing embeddings to file")
        return
    print("Writing embedding to file{}".format(outFile))
    imgs_count = len(file_ids[0:steps*batch])
    print("total image names: ", len(file_ids[0:steps*batch]))
    timestart = time.time()
    with h5py.File(outFile, 'w') as hf:
        img_data = hf.create_dataset("vectors",shape=(imgs_count,160,160,3), dtype ='uint8',chunks=True
                                     ,compression="gzip",compression_opts=9)
        #write the ids at the end
        #img_ids = hf.create_dataset("image_names", shape=(total_imgs,namebytesz))
        for idx in tqdm(range(steps)):
            data=  get_testImages_fast( batch=batch,file_List=file_list, file_Ids=file_ids,currIndx= idx)
            #data[0] is image
            vectors = np.ascontiguousarray(data[0], dtype='uint8')
            #image_names =data[1]
            #image_names = np.array([bytes(name, "ascii") for name in image_names ])
            startidx = idx*batch
            stopidx= (idx+1)*batch
            img_data[startidx:stopidx, :, :] = vectors
            #img_ids[idx:(idx + 1) * batch:, :, :] = image_names
        image_names = np.array([bytes(name, "ascii") for name in file_ids[0:imgs_count] ])
        img_ids = hf.create_dataset("image_names", data=image_names)
    timestop = time.time()
    print("Time for generation {} mins".format((timestop - timestart)/60))

def save_QueryImagesAsHdf5():

    img_dir = "D:/prjs/im-similarity/data/query"
    #img_dir = 'C:/Users/parajav/PycharmProjects/isc/query' './list_files/subset_1_queries'
    image_list, ids = getImIds('./list_files/dev_queries', img_dir)#'./list_files/dev_queries'
    print("total images = ", len(ids))
    #print("calling fast method")
    #gen_imageEmbeddingsSlow(image_list, ids, outFile='./image_qry_zip.hdf5',bs=32)
    print("calling other method")
    gen_imageEmbeddings(image_list, ids, outFile='./image_dev_queries.hdf5', bs=50)
    #print("calling chunks")
    #gen_imageEmbeddingsChunks(image_list, ids, outFile='./image_qry_chunk.hdf5', bs=32)

def save_subsetImagesAsHdf5():
    img_dir = "D:/prjs/im-similarity/data/query"
    #img_dir = 'C:/Users/parajav/PycharmProjects/isc/query' './list_files/subset_1_queries'
    image_list, ids = getImIds('./list_files/subset_1_queries', img_dir)#'./list_files/dev_queries'
    print("**********genereating query images as hdf5")
    gen_imageEmbeddings(image_list, ids, outFile='./data/image/im_subset_query.hdf5', bs=16)
    image_list, ids = getImIds('./list_files/subset_1_references', "C:/Users/parajav/PycharmProjects/isc/reference/reference")
    print("************genereating ref images as hdf5")
    gen_imageEmbeddings(image_list, ids, outFile='./data/image/im_subset_ref.hdf5', bs=16)

def save_RefImagesAsHdf5():
    '''
    anchor_img_dir = "D:/prjs/im-similarity/data/query"
    q_image_list, q_ids = getImIds('./list_files/dev_queries', anchor_img_dir)
    gen_imageEmbeddings(q_image_list, q_ids, outFile='./image_querydev_small.hdf5', )
    print(len(q_image_list))
    '''

    img_dir = "C:/Users/parajav/PycharmProjects/isc/reference/reference"
    image_filepaths, ids = getImIds('./list_files/references', img_dir)
    print("total images = " , len(ids))
    #gen_imageEmbeddings(image_filepaths, ids, outFile='./image_full_ref_small.hdf5',bs=32 )
    batchsize=50
    interval = 50000  # 50K
    print("Interval= {}, batchsize={}, imagesperbatch ={}".format(interval, batchsize, interval/batchsize))
    iters = int(len(ids) / interval)
    print("iters", iters)
    for ii in range(9, iters):
        i0 = ii * interval
        i1 = (ii + 1) * interval
        print("reading files from {} to {}".format(i0, i1))
        gen_imageEmbeddings(file_list=image_filepaths[i0:i1], file_ids=ids[i0:i1],
                            outFile='./data/image_full_ref_' + str(ii) + '.hdf5', bs= batchsize)
    print("completed")

def save_ExtendedRefImagesAsHdf5():

    #we will use R_List and ref_img_dir  and REF_IDS
    from isc.io import read_predictions_as_arrays
    pred_q, pred_r = read_predictions_as_arrays('./data/submission/s5_full_TH0.98.csv')
    pred_r_fullfile = ['D:\\prjs\\im-similarity\\data\\subset\\reference' + '/' + curr +'.jpg' for curr in pred_r]
    # Tryign to remove duplicates using set of unique loses the original sequence order
    #pred_r_fullfile = list(set(pred_r_fullfile))
    pred_r_fullfile.extend(item for item in R_List if item not in pred_r_fullfile)
    # generate the IDS from  fullfilename
    #the last [:-4] is for removing '.jpg'
    combinedIDs = [
        name.split('/')[-1][:-4]
        for name in pred_r_fullfile
    ]
    #gen_imageEmbeddings(pred_r_fullfile, combinedIDs, outFile='./data/image/image_extended_Ref.hdf5', bs=25)
    filepath = './list_files/subset_ref_extended'
    with open(filepath, "w") as pfile:
        #pfile.write("query_id,reference_id,score\n")
        for p in combinedIDs:
            row = f"{p}"
            pfile.write(row + "\n")
    #copy these image to some location as well
    '''
    import shutil, os
    dest_dir = 'D:\\prjs\\im-similarity\\data\\subset\\reference'
    if not os.path.exists(dest_dir):
        print("creating a directory as it doesnt exists")
        os.mkdir(dest_dir)
    for f in pred_r_fullfile:
        shutil.copy(f,dest_dir )
    '''

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
    k = 1
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
    print("name= {}: min={},max={}, mean={},median={},"
          " type={}, shape={}".format(str, np.min(img), np.max(img),
                                      np.mean(img),np.median(img)
                                      ,img.dtype, img.shape))

def test_generator():
    train_generator = generate_triplets(start=0, stop=100,BATCH_SIZE=1)
    data = next(train_generator)
    [A, P, N], label = data
    idx=0
    plot_triplets([A[idx],P[idx],N[idx]])
    getMinMax(A[idx], "A")
    getMinMax(P[idx], "P")
    getMinMax(N[idx], "N")

def analyze_subsetAcc():
    query_image_ids, XQ = read_descriptors(['./data/subset_query_resnet50Reg0.8.hdf5'])
    ref_image_ids, XD = read_descriptors(['./data/subset_ref_resnet50Reg0.8.hdf5'])
    d = 256
    index = faiss.IndexFlatL2(d)
    k = 1
    index.add(XD)
    D, I = index.search(XQ, k)  # search top k
    print("matching index after training....")
    getMatchingScore(I, k)
    #some prototype
    # print(I)
    lenI = len(I)
    matches = [index for index, value in enumerate(I) if value[0] == index]
    non_matching = [i for i in range(lenI) if i not in matches]
    DTrue = D[matches]
    DFalse = D[non_matching]
    getMinMax(DTrue, 'matching')
    getMinMax(DFalse, 'non-matching')
    index = [True if i in matches else False for i in range(lenI)]# idnex of mathcing
    #Inew = [I[i] if idx == True else False for idx in index]
    Dth =1.0
    matches_new = [index for index, value in enumerate(I) if value[0] == index and D[index] < Dth]
    getMatchingScore(I, k)

def mytest_hdf5loader():
    myHd5File = './data/image/im_subset_query.hdf5'
    hdf5loader = Hdf5Sequence(myHd5File, idlist='', batch_size=2)
    batchdata= hdf5loader[0]
    batchdata1 = hdf5loader[1]
    plot_batches(batchdata)

def calculate_faissmetrics_inference():
    from isc.metrics import GroundTruthMatch, PredictedMatch
    from isc.io import read_ground_truth
    from isc.descriptor_matching import knn_match_and_make_predictions
    import pickle
    from models.resnet50Reg import embedding_model
    from isc.metrics import to_arrays
    gt_matches = read_ground_truth('./list_files/subset_1_ground_truth.csv')
    I0 = pickle.load('./data/L2Index_2_prev.p')
    base_model = embedding_model()
    # base_model.load_weights("./models/weights/" + "resnet50goodEmbeddings_86_.hdf5")
    base_model.load_weights("./models/weights/" + "OFF_Em_res50_best.hdf5")
    # findAccuracy(base_model, save=False) #uses old method -------->getMatchingScore(I, k)
    qimage_list, qids = getImIds('./list_files/subset_1_queries', "D:/prjs/im-similarity/data/query")
    xq = gen_embeddingsSeq(base_model, './data/image/im_subset_query.hdf5', qids, outFile='', batch=50)
    rimage_list, rids = getImIds('./list_files/subset_1_references',
                               "C:/Users/parajav/PycharmProjects/isc/reference/reference")
    xd = gen_embeddingsSeq(base_model, './data/image/im_subset_ref.hdf5', rids,
                           outFile='./data/embed/subset_ref_em_resnet50Reg.hdf5', batch=50)
    predictions = knn_match_and_make_predictions(
        xq, qids, xd, rids, 1, metric=faiss.METRIC_L2 )
    #load dump file
    #use old method "findAccuracy" to verify
    y_true, probas_pred = to_arrays(gt_matches, predictions)

