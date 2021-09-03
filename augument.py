import albumentations as A
from settings import *
import numpy as np

imsize=IM_SIZE[0]
augHndl =[]
augHndl1 = A.Compose([
    A.Blur(blur_limit=7),
    A.HorizontalFlip(p=0.5),#(-0.9, 1.2)
    #albu.Normalize(),
    A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3, brightness_by_max=True),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(),
    A.RandomSnow(brightness_coeff=1.5, p=0.5),
    #
    A.RandomSizedCrop((imsize - 50, imsize - 1), imsize, imsize)
    ])
augHndl2 = A.Compose([
    A.ElasticTransform(p=0.5),
    A.ISONoise(),
    #
    A.RandomSizedCrop((imsize - 50, imsize - 1), imsize, imsize)
])
augHndl3 = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.7),
    A.Flip(p=0.5),
    A.ISONoise(),
    A.RandomSizedCrop((imsize - 50, imsize - 1), imsize, imsize)
])

def get_augument_image_internal(image):
    choice = np.random.choice([1,2,3])
    #choice=1 #for test
    #print("choice= ", choice)
    hndl=augHndl1
    if choice==1:
        hndl = augHndl1
    elif choice ==2:
        hndl = augHndl2
    else:
        hndl = augHndl3
    augmented = hndl(image=image)
    return augmented['image']

def add_noise(image):
    noise_typ = np.random.choice(["gauss","s&p"])
    #noise_typ = "gauss" #"gauss"
    image = np.squeeze(image)
    image = image/255
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      #print("input image mean = ",np.mean(image))
      var = 0.0059
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      #print("noisy image mean = ", np.mean(noisy))
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      noisy = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      noisy[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      noisy[coords] = 0
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
    return np.uint8(noisy*255)


def apply_augumentaion_wrapper(img):
    return get_augument_image_internal(np.squeeze(img))