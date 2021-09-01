# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import numpy as np
import faiss
from faiss.contrib import exhaustive_search
import logging

import cv2

from .metrics import PredictedMatch

def query_iterator(xq):
    """ produces batches of progressively increasing sizes """
    nq = len(xq)
    bs = 32
    i = 0
    while i < nq:
        xqi = xq[i : i + bs]
        yield xqi
        if bs < 20000:
            bs *= 2
        i += len(xqi)

#########################
# These two functions are there because current Faiss contrib
# does not proporly support IP search
#########################


def threshold_radius_nres_IP(nres, dis, ids, thresh):
    """ select a set of results """
    mask = dis > thresh
    new_nres = np.zeros_like(nres)
    o = 0
    for i, nr in enumerate(nres):
        nr = int(nr)   # avoid issues with int64 + uint64
        new_nres[i] = mask[o : o + nr].sum()
        o += nr
    return new_nres, dis[mask], ids[mask]


def apply_maxres_IP(res_batches, target_nres):
    """find radius that reduces number of results to target_nres, and
    applies it in-place to the result batches used in range_search_max_results"""
    alldis = np.hstack([dis for _, dis, _ in res_batches])
    alldis.partition(len(alldis) - target_nres)
    radius = alldis[-target_nres]

    LOG = logging.getLogger(exhaustive_search.__name__)

    if alldis.dtype == 'float32':
        radius = float(radius)
    else:
        radius = int(radius)
    LOG.debug('   setting radius to %s' % radius)
    totres = 0
    for i, (nres, dis, ids) in enumerate(res_batches):
        nres, dis, ids = threshold_radius_nres_IP(nres, dis, ids, radius)
        totres += len(dis)
        res_batches[i] = nres, dis, ids
    LOG.debug('   updated previous results, new nb results %d' % totres)
    return radius, totres

def search_with_capped_res(xq, xb, num_results, metric=faiss.METRIC_L2):
    """
    Searches xq into xb, with a maximum total number of results
    """
    index = faiss.IndexFlat(xb.shape[1], metric)
    index.add(xb)
    # logging.basicConfig()
    # logging.getLogger(exhaustive_search.__name__).setLevel(logging.DEBUG)

    if metric == faiss.METRIC_INNER_PRODUCT:
        # this is a very ugly hack because contrib.exhaustive_search does
        # not support IP search correctly. Do not use in a multithreaded env.
        apply_maxres_saved = exhaustive_search.apply_maxres
        exhaustive_search.apply_maxres = apply_maxres_IP

    radius, lims, dis, ids = exhaustive_search.range_search_max_results(
        index, query_iterator(xq),
        1e10 if metric == faiss.METRIC_L2 else -1e10,      # initial radius does not filter anything
        max_results=2 * num_results,
        min_results=num_results,
        ngpu=-1   # use GPU if available
    )

    if metric == faiss.METRIC_INNER_PRODUCT:
        exhaustive_search.apply_maxres = apply_maxres_saved

    n = len(dis)
    nq = len(xq)
    if n > num_results:
        # crop to num_results exactly
        if metric == faiss.METRIC_L2:
            o = dis.argpartition(num_results)[:num_results]
        else:
            o = dis.argpartition(len(dis) - num_results)[-num_results:]
        mask = np.zeros(n, bool)
        mask[o] = True
        new_dis = dis[mask]
        new_ids = ids[mask]
        nres = [0] + [
            mask[lims[i] : lims[i + 1]].sum()
            for i in range(nq)
        ]
        new_lims = np.cumsum(nres)
        lims, dis, ids = new_lims, new_dis, new_ids

    return lims, dis, ids


def match_and_make_predictions(xq, query_image_ids, xb, db_image_ids, num_results, ngpu=-1, metric=faiss.METRIC_L2):
    lims, dis, ids = search_with_capped_res(xq, xb, num_results, metric=metric)
    nq = len(xq)

    if metric == faiss.METRIC_L2:
        # use negated distances as scores
        dis = -dis

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[ids[j]],
            dis[j]
        )
        for i in range(nq)
        for j in range(lims[i], lims[i + 1])
    ]
    return predictions


def knn_match_and_make_predictions(xq, query_image_ids, xb, db_image_ids, k, ngpu=-1, metric=faiss.METRIC_L2):

    if faiss.get_num_gpus() == 0 or ngpu == 0:
        D, I = faiss.knn(xq, xb, k, metric)
    else:
        d = xq.shape[1]
        index = faiss.IndexFlat(d, metric)
        index.add(xb)
        index = faiss.index_cpu_to_all_gpus(index)
        D, I = index.search(xq, k=k)
    nq = len(xq)

    if metric == faiss.METRIC_L2:
        # use negated distances as scores
        D = -D

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[I[i, j]],
            D[i, j]
        )
        for i in range(nq)
        for j in range(k)
    ]
    return predictions




def range_result_read(fname):
    """ read the range search result file format """
    f = open(fname, "rb")
    nq, total_res = np.fromfile(f, count=2, dtype="int32")
    nres = np.fromfile(f, count=nq, dtype="int32")
    assert nres.sum() == total_res
    I = np.fromfile(f, count=total_res, dtype="int32")
    return nres, I

def orb_single_match(descriptors_train, descriptors_query):
    DEBUG =False
    ORB_distThresh = 60
    NUM_MATCHES_TH = 35
    # define %GOOD match
    ORB_ResultThresh = 0.50  # >50% match  is good
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_train, descriptors_query)
    num_matches = len(matches)
    if num_matches>=NUM_MATCHES_TH:
            # The matches with shorter distance are the ones we want. So, we sort the matches according to distance
        matches = sorted(matches, key=lambda x: x.distance)
        good = []
        for m in matches:
            if m.distance < ORB_distThresh:
                good.append([m])
        nGoodMatches = len(good)
        score = nGoodMatches / len(matches)
        if DEBUG:
            # Print total number of matching points between the training and query images
            print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
            print("\nNumber of GOOD Matches Between The Training and Query Images: ", len(good))
            print(" ratio of good matches = ", 100 * score)
        isMatch = (nGoodMatches / len(matches) > ORB_ResultThresh)
    else:
        isMatch = False
        score = 0
    #Score is in %(1.0->total match; 0-> no match)
    #negate the score as lowe score means MATCH
    score = 1.0-score# no this is dist
    return score,isMatch

def orb_match_and_make_predictions(xq, query_image_ids, xb, db_image_ids,Max_Matches=1):
    #D, I = faiss.knn(xq, xb, k, metric)
    num_QImages =len(xq)
    num_dbImages = len(xb)
    counter=0 #counter to track no of matches found
    Dist = np.zeros((num_QImages,Max_Matches))
    DB_match_ids = np.zeros((num_QImages, Max_Matches),dtype='int')
    DB_match_ids[:,:] =-1 # initialize all to -1
    for qi in range(num_QImages):
        counter =0
        if qi < 100 or qi % 100 == 0:
            print(" Finding match for Query # {}, imagename{}".format(qi, query_image_ids[qi]))#,end="\r", flush=True)
        #print(f"processing {qi}/{len(num_QImages)} {query_image_ids[qi]}", end="\r", flush=True)
        for di in range(num_dbImages):
            if di in DB_match_ids:#dont compare with already found Reference
                continue;
            try:
                currScore, isMatch =orb_single_match(xb[di],xq[qi])
                if isMatch:
                    print("matched Query # {}, with Refernce #{} ".format(qi, di))
                    Dist[qi,counter] = currScore
                    DB_match_ids[qi,counter] = di
                    counter =counter+1
                if counter>=Max_Matches:
                    break#break out of inner loop
            except:
                print("*********Empty feature found,do nothing and move on to next")
        #print("found {} matches for image{}".format(counter, qi))
    print("found all matches")

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[DB_match_ids[i,j]],
            Dist[i, j]
        )
        for i in range(num_QImages)
        for j in range(Max_Matches)
    ]
    return predictions

def norm_vec(x):
    output = np.sqrt(max(sum(x**2), 1e-4))
    return output
def cosine_similarity(a,b):
    out = sum(np.multiply(a,b)/(norm_vec(a)*norm_vec(b) ))
    return out

def net_match_and_make_predictions(xq, query_image_ids, xb, db_image_ids,Max_Matches=1):
    #D, I = faiss.knn(xq, xb, k, metric)
    num_QImages =len(xq)
    num_dbImages = len(xb)
    counter=0 #counter to track no of matches found
    Dist = np.zeros((num_QImages,Max_Matches))
    DB_match_ids = np.zeros((num_QImages, Max_Matches),dtype='int')
    DB_match_ids[:,:] =-1 # initialize all to -1
    for qi in range(num_QImages):
        counter =0
        if qi < 100 or qi % 100 == 0:
            print(" Finding match for Query # {}, imagename{}".format(qi, query_image_ids[qi]))#,end="\r", flush=True)
        #print(f"processing {qi}/{len(num_QImages)} {query_image_ids[qi]}", end="\r", flush=True)
        for di in range(num_dbImages):
            if di in DB_match_ids:#dont compare with already found Reference
                continue;
            try:
                currScore=cosine_similarity(xb[di],xq[qi])
                if currScore>0.5:
                    print("matched Query # {}, with Refernce #{} ".format(qi, di))
                    Dist[qi,counter] = currScore
                    DB_match_ids[qi,counter] = di
                    counter =counter+1
                if counter>=Max_Matches:
                    break#break out of inner loop
            except:
                print("*********Empty feature found,do nothing and move on to next")
        #print("found {} matches for image{}".format(counter, qi))
    print("found all matches")

    predictions = [
        PredictedMatch(
            query_image_ids[i],
            db_image_ids[DB_match_ids[i,j]],
            Dist[i, j]
        )
        for i in range(num_QImages)
        for j in range(Max_Matches)
    ]
    return predictions