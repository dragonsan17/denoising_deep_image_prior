#utils
import torch
import torch.nn as nn
import torchvision
import sys
import cv2
import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

def crop_image(img, d=32):
    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def plot_image_grid(images_np, nrow =8, factor=1, interpolation='lanczos', labels = None):
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    fig = plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    columns = len(images_np)
    rows = 1
    for i in range(columns):
        img = images_np[i]
        axes = fig.add_subplot(rows, columns, i+1)
        plt.imshow(img.transpose(1, 2, 0), interpolation=interpolation)
        if(labels!=None):
          axes.set_title(labels[i])
    plt.show()
    
def load(path):
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np

def fill_noise(x, noise_type):
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, spatial_size, noise_type='u', var=1./10):
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)
    
    fill_noise(net_input, noise_type)
    net_input *= var
        
    return net_input

def pil_to_np(img_PIL):
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()[0]

def noisy(noise_typ,image, mean = 0, sigma = 50, s_vs_p = 0.5, amount = 0.05):
    if noise_typ == "gauss":
      row,col,ch= image.shape
      gauss = np.random.normal(mean,sigma,(row,col,ch)).astype(float)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return np.clip(noisy,0,1).astype(np.float32)
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 1.5 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return np.clip(noisy,0,1).astype(np.float32)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        noise = np.random.normal(0, sigma,
                                 image.shape)
        noisy = image + image * noise
        return np.clip(noisy,0,1).astype(np.float32)
