import os
import keras
import h5py
import math
import numpy as np
import datetime as dt
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io



class Hdf5Sequence(keras.utils.Sequence):

    def __init__(self, myHd5File, idlist, batch_size):
        self.srcFile, self.Ids = myHd5File, idlist
        self.batch_size = batch_size
        self.hh = h5py.File(myHd5File, "r")  # call these only once
        self.data_size = self.hh["vectors"].shape[0]  #
        self.size_per_batch = self.data_size / batch_size
        self.img_size = (160, 160)  # always for this case/project

    def __len__(self):
        return math.ceil(self.data_size / self.batch_size)

    def my_preprocess_img(self, img):
        return img

    def __getitem__(self, idx):
        """Return tuple(input, id) or (img, id) correspondidng to batch #idx
                single Call to getitem will return batch_size length of data"""
        # start = current * size_per_batch
        # end = (current + 1) * size_per_batch
        startIndex = idx * self.batch_size
        stopIndex = startIndex + self.batch_size
        descs = []
        names = []
        descs.append(np.array(self.hh["vectors"][startIndex:stopIndex]))
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

def my_test_size():
    img_dir = "C:/Users/parajav/PycharmProjects/isc/reference/reference"
    image_fullnames = glob(os.path.join(img_dir, "*.jpg"))
    img_path = image_fullnames[0]
    print("imagename: ", img_path)
    print("****************case1: read and save file as numpy array")
    save_path = '../numpy.hdf5'
    imgsz=os.path.getsize(img_path)
    print('image size: %d bytes' % imgsz)
    hf = h5py.File(save_path, 'w')  # open a hdf5 file
    start = dt.datetime.now()
    img_np = np.array(Image.open(img_path))
    dset = hf.create_dataset('default', data=img_np)  # write the data to hdf5 file
    hf.close()  # close the hdf5 file
    end = dt.datetime.now()
    print(" total time,  ", (end - start).microseconds/1000, "milliseconds")
    print('hdf5 file size: %d bytes; ratio: hdf5/origImg %f' % (os.path.getsize(save_path),
                                                   os.path.getsize(save_path)/imgsz))
    print("****************case2: read and save file as numpy array(uint8)")
    save_path = '../numpyuint8.hdf5'
    imgsz = os.path.getsize(img_path)
    print('image size: %d bytes' % imgsz)
    hf = h5py.File(save_path, 'w')  # open a hdf5 file
    start = dt.datetime.now()
    img_np = np.array(Image.open(img_path),dtype='uint8')
    dset = hf.create_dataset('default', data=img_np)  # write the data to hdf5 file
    hf.close()  # close the hdf5 file
    end = dt.datetime.now()
    print(" total time,  ", (end - start).microseconds/1000, "milliseconds")
    print('hdf5 file size: %d bytes; ratio: hdf5/origImg %f' % (os.path.getsize(save_path),
                                                                os.path.getsize(save_path) / imgsz))


    print("****************case3: read and save file as python binary file")
    save_path = '../test.hdf5'
    hf = h5py.File(save_path, 'w')  # open a hdf5 file
    start = dt.datetime.now()
    with open(img_path, 'rb') as img_f:
        binary_data = img_f.read()  # read the image as python binary
    binary_data_np = np.asarray(binary_data)
    dset = hf.create_dataset('default', data=binary_data_np)  # write the data to hdf5 file
    hf.close()  # close the hdf5 file
    end = dt.datetime.now()
    print(" total time,  ", (end - start).microseconds/1000, "milliseconds")
    print('hdf5 file size: %d bytes; ratio: hdf5/origImg %f' % (os.path.getsize(save_path),
                                                                os.path.getsize(save_path) / imgsz))
    print("****************case3b: read and save resized file as python binary file")
    save_path = '../test3b.hdf5'
    hf = h5py.File(save_path, 'w')  # open a hdf5 file
    start = dt.datetime.now()
    with open(img_path, 'rb') as img_f:
        binary_data = img_f.read()  # read the image as python binary
    #binary_data_np = np.asarray(binary_data)
    im = Image.open(io.BytesIO(binary_data))
    resizedImage = im.resize((160, 160))
    buf = io.BytesIO()
    #resizedImage.save(buf, format='JPEG')
    dset = hf.create_dataset('default', data=np.asarray(resizedImage))  # write the data to hdf5 file
    hf.close()  # close the hdf5 file
    end = dt.datetime.now()
    #to retrieve data use the following:
    #hf = h5py.File(hdf5_file, 'r')  # open a hdf5 file
    #data = np.array(hf[key])  # write the data to hdf5 file
    #img = Image.open(io.BytesIO(data))
    #byte_im = buf.getvalue()
    print(" total time,  ", (end - start).microseconds / 1000, "milliseconds")
    print('hdf5 file size: %d bytes; ratio: hdf5/origImg %f' % (os.path.getsize(save_path),
                                                                os.path.getsize(save_path) / imgsz))

    print("****************case4: opencv2 img with uint8")
    save_path = '../testopencv2.hdf5'
    hf = h5py.File(save_path, 'w')  # open a hdf5 file
    start = dt.datetime.now()
    image = np.array(cv2.imread(img_path))
    #binary_data_np = np.asarray(binary_data)
    vectors = np.ascontiguousarray(image, dtype='uint8')
    dset = hf.create_dataset('default', data=vectors)  # write the data to hdf5 file
    hf.close()  # close the hdf5 file
    end = dt.datetime.now()
    print(" total time,  ", (end - start).microseconds/1000, "milliseconds")
    print('hdf5 file size: %d bytes; ratio: hdf5/origImg %f' % (os.path.getsize(save_path),
                                                                os.path.getsize(save_path) / imgsz))
    print("image shape from opencv: ",image.shape )
    print("****************case5: opencv2 img with resize")
    save_path = '../testopencv2resized.hdf5'
    hf = h5py.File(save_path, 'w')  # open a hdf5 file
    start = dt.datetime.now()
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160,160), interpolation=cv2.INTER_AREA)
    # binary_data_np = np.asarray(binary_data)
    vectors = np.ascontiguousarray(image, dtype ='uint8')
    dset = hf.create_dataset('default', data=vectors)#,shape=(160,160,3),
                #maxshape=(160,160,3),compression="gzip",compression_opts=9)
    hf.close()  # close the hdf5 file
    end = dt.datetime.now()
    print("image type: ", image.dtype)
    print(" total time,  ", (end - start).microseconds / 1000, "milliseconds")
    print('hdf5 file size: %d bytes; ratio: hdf5/origImg %f' % (os.path.getsize(save_path),
                                                                os.path.getsize(save_path) / imgsz))
    print("image shape from opencv: ", image.shape)


my_test_size()