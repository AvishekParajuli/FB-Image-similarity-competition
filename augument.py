import albumentations as A
from settings import *
import numpy as np
import augly.image as imaugs
import augly.utils as utils
from augly.image import (aug_np_wrapper, overlay_emoji, scale, random_noise, color_jitter)
from augly.image import overlay_stripes,overlay_text,pad,pad_square,pixelization, meme_format
from augly.image import masked_composite,meme_format,opacity,overlay_emoji,overlay_image,overlay_onto_screenshot
import os
import cv2

imsize=IM_SIZE[0]
augHndl =[]
augHndl1 = A.Compose([
    A.Blur(blur_limit=7),
    A.HorizontalFlip(p=0.5),#(-0.9, 1.2)
    #albu.Normalize(),
    A.RandomBrightnessContrast(contrast_limit=0.3, brightness_limit=0.3),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(),
    #A.RandomSnow(brightness_coeff=1.5, p=0.5),
    #
    A.RandomSizedCrop((imsize - 50, imsize - 1), imsize, imsize)
    ])
augHndl2 = A.Compose([
    #A.ElasticTransform(p=0.5),
    A.ColorJitter(p=0.5),
    A.GaussNoise(),
    A.Resize( imsize, imsize,3)
    #A.RandomSizedCrop((imsize - 30, imsize - 30), imsize, imsize)
])
augHndl3 = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.7),
    A.Flip(p=0.5),
    A.GaussNoise(),
    A.RandomSizedCrop((imsize - 50, imsize - 1), imsize, imsize)
])

def get_augument_image_internal(image):
    choice = np.random.choice([1,2,2,2,3])
    #choice=2 #for test
    #print("choice= ", choice)
    hndl=augHndl1
    if choice==1:
        hndl = augHndl1
    elif choice ==2:
        hndl = augHndl2
        image = apply_aug_internal(image)
        #print("interim-img-after augly: ", image.shape, image.dtype, np.max(image), np.min(image))
    else:
        hndl = augHndl3
    augmented = hndl(image=image)
    outimg = augmented['image']
    if outimg.shape[-1]==4:
      outimg = cv2.cvtColor(outimg, cv2.COLOR_RGBA2RGB)
    #print("outimg: ",outimg.shape, outimg.dtype, np.max(outimg), np.min(outimg))
    return outimg

import numpy as np

def apply_aug_internal(img, choice=-1):
  selected = np.random.randint(0,10)
  if choice!=-1:
    selected=choice
  #print("Augly current choice: ", selected)
  aug_img = np.array((1,1))
  rnd_1_to_5 = np.random.randint(1,5)
  #print("rnd_1_to_5: ",rnd_1_to_5)
  if selected==0:
    aug_img = aug_np_wrapper(img,overlay_emoji, **{'opacity': 0.7, 'y_pos': 0.45, 'emoji_size':0.13*rnd_1_to_5} )
    aug_img= cv2.cvtColor(aug_img, cv2.COLOR_RGBA2RGB)
  elif selected ==1:
    aug_img = aug_np_wrapper(img,pixelization,**{'ratio':0.1*rnd_1_to_5})
  elif selected ==2:
    aug_img = aug_np_wrapper(img,overlay_text )
  elif selected ==3:
    aug_img = aug_np_wrapper(img,overlay_stripes,**{'line_width':0.13*rnd_1_to_5,'line_angle':10*(rnd_1_to_5-1)}  )
  elif selected ==4:
    aug_img = aug_np_wrapper(img,meme_format  , **{'caption_height':17*rnd_1_to_5,'meme_bg_color':(0, 0, 0), 'text_color':(255,255,255)} )
  elif selected ==5:
    aug_img = aug_np_wrapper(img, pad_square)
  else:
    aug_img = aug_np_wrapper(img,overlay_onto_screenshot  , template_filepath=os.path.join(utils.SCREENSHOT_TEMPLATES_DIR, "mobile.png") )
  if aug_img.shape[-1]==4:
      aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGBA2RGB)
  
  return aug_img

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
  #return img
  return get_augument_image_internal(np.squeeze(img))