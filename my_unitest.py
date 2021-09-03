import augument as myaug
from loader.fb_image_gen_pre import plot_triplets,plt,read_image, get_triplet, Hdf5Sequence,plot_batches
from settings import *
from utils import getMinMax
import numpy as np


def plot_images(imlist):
    imlen= len(imlist)
    plt.figure(figsize=(6, 2))
    for i in range(imlen):
        plt.subplot(1, imlen,i+1)
        plt.imshow(imlist[i])
        if i==0:
            plt.title("original")
        else:
            plt.title("augumented")
    plt.show()

def mytest_augumentation():
    data = get_triplet(0, mode ='train')
    plot_triplets(data)# this will only add noise

    #test1:
    idx=0
    a = read_image(Q_List[idx])
    aug_im = myaug.apply_augumentaion_wrapper(a)
    getMinMax(a,"original")
    getMinMax(aug_im, "augumented")
    plot_images([a, aug_im])
    #for i in range(10):
        #plot_images([a, myaug.apply_augumentaion_wrapper(a)])
    transform1 = myaug.A.Compose([
        myaug.A.RandomBrightnessContrast(contrast_limit=0.3,brightness_limit=0.3,brightness_by_max=True, p=1.0)
    ])
    transform2 = myaug.A.Compose([
        myaug.A.ElasticTransform(p=1.0 )
    ])
    transform3 = myaug.A.Compose([
        myaug.A.RandomSnow(p=1.0, brightness_coeff=1.5)
    ])#inverted type
    transform4 = myaug.A.Compose([
        myaug.A.RandomGridShuffle(p=1.0,grid=(1,1))
    ])#lower grid size(default also good)
    '''transform5 = myaug.A.Compose([
        myaug.A.RandomSunFlare(p=1.0,src_color=(50,60,80),
                                  num_flare_circles_lower=1, num_flare_circles_upper=6)
    ])#redice it
    '''
    transform5 = myaug.A.Compose([
        myaug.A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=1.0)
        ])# lower grid size(default also good)
    transform6 = myaug.A.Compose([
        myaug.A.RandomToneCurve(p=1.0,scale =0.3)
    ])
    b1 = transform1(image=a)["image"]
    b2 = transform2(image=a)["image"]
    b3 = transform3(image=a)["image"]
    b4 = transform4(image=a)["image"]
    b5 = transform5(image=a)["image"]
    b6 = transform6(image=a)["image"]
    plot_images([a, b1,b2,b3])
    plot_images([a,b4,b5,b6])



def mytest_hdf5loader():
    myHd5File = './data/image/im_subset_query.hdf5'
    hdf5loader = Hdf5Sequence(myHd5File, idlist='', batch_size=2)
    batchdata= hdf5loader[0]
    batchdata1 = hdf5loader[1]
    plot_batches(batchdata)

def main():
    mytest_augumentation()
    #mergeHdf5Files()

def dummy():
    import h5py
    import os
    d_names = ['./data/image/image0.hdf5', './data/image/image1.hdf5']
    d_struct = {}  # Here we will store the database structure
    for i in d_names:
        f = h5py.File(i, 'r+')
        print("filename: ", i)
        d_struct[i] = f.keys()
        #print("keys: ",d_struct[i])
        f.close()

    for i in d_names:
        for j in d_struct[i]:
            os.system('h5copy -i %s -o output.h5 -s %s -d %s' % (i, j, j))
def mergeHdf5Files():
    import h5py
    import os
    d_names = ['./data/image/image_extended_Ref.hdf5', './data/image/image_full_ref_0.hdf5',
               './data/image/image_full_ref_1.hdf5','./data/image/image_full_ref_2.hdf5']
    outfilename= './data/image/mergedRefExtended_0to2.hdf5'
    print("creating merged filename with name: ", outfilename)
    with h5py.File(outfilename, mode='w') as h5fw:
        row1 = 0
        file_ids =[]
        for h5name in d_names:
            h5fr = h5py.File(h5name, 'r')
            dset1 = list(h5fr.keys())[1]# 1->vectors; 2->image_names
            #arr_data = h5fr['vectors'][:]
            dslen = h5fr['vectors'].shape[0]
            dsshape = h5fr['vectors'].shape
            if row1 == 0:
                h5fw.create_dataset('vectors', dtype='uint8', shape=dsshape, maxshape=(None, 160,160,3))
            if row1 + dslen <= len(h5fw['vectors']):
                h5fw['vectors'][row1:row1 + dslen, :] = h5fr['vectors']#[:]
                #im_names= np.array(myfile["image_names"][:]).astype(str).tolist()
            else:
                h5fw['vectors'].resize((row1 + dslen, 160,160,3))
                h5fw['vectors'][row1:row1 + dslen, :,:] = h5fr['vectors']#[:]
            row1 += dslen
            im_names = np.array(h5fr["image_names"][:]).astype(str).tolist()
            file_ids.extend(im_names)
        image_names = np.array([bytes(name, "ascii") for name in file_ids])
        h5fw.create_dataset("image_names", data=image_names)
    print("========completeing writing merged file")


if __name__ == '__main__':
    main()