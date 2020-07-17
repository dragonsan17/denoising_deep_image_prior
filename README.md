# Deep Image Prior - Denoising Images

This is an implementation of __Deep Image Prior__, specifically for denoising CIFAR - 10 images. This approach is based on the idea that a randomly initialized encoder-decoder architecture can be used as an image prior for standard image inversion tasks such as denoising. This requires no training and no specific knowledge of the underlying noise model.

The __*'denoising_single_image.ipynb'*__ jupyter notebook shows inference of this approach on sample CIFAR-10 images and an F16-GT Airplane image (used in the original paper).

The __*'denoising_f16gt.ipynb'*__ jupyter notebook shows inference of this approach on an F16-GT Airplane image (used in the original paper), on all types of noises with the same configuration. This is to show that this approach can work without knowing anything about the underlying model used for inducing noise.

There are separate implementations of various types of noises such as Gaussian, Poisson, Salt and Pepper and Speckle noise present inside the 'utils.py' file.

__Paper__ : https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf

## Result on F16-GT image (used in the original implementation)

* __Output__ : Final output by the Network
* __Average Output__ : Output on last iterations averaged using exponential sliding window

*As reported in the paper, Average Output of the model gives excellent results in __Blind image denoising__.*

* Gaussian Noise : 
![GitHub Logo](/results/gaussian_f16gt.PNG)
* Salt and Pepper Noise : 
![GitHub Logo](/results/salt_and_pepper_f16gt.PNG)
* Poisson Noise : 
![GitHub Logo](/results/poisson_f16gt.PNG)
* Speckle Noise : 
![GitHub Logo](/results/speckle_f16gt.PNG)

## Some Results on CIFAR-10 Images:

* __Output__ : Final output by the Network
* __Average Output__ : Output on last iterations averaged using exponential sliding window

*As reported in the paper, Average Output of the model gives excellent results in __Blind image denoising__.*

* Gaussian Noise : 
![GitHub Logo](/results/gaussian_cifar.PNG)
* Salt and Pepper Noise : 
![GitHub Logo](/results/salt_and_pepper_cifar.PNG)
* Poisson Noise : 
![GitHub Logo](/results/poisson_cifar.PNG)
* Speckle Noise : 
![GitHub Logo](/results/speckle_cifar.PNG)
