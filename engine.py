import matplotlib.pyplot as plt
%matplotlib inline
import torch.nn as nn
import os
import numpy as np
import torch
import torch.optim
from skimage.measure import compare_psnr
from tqdm.notebook import tqdm
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
from utils import *
from hourglass_architecture import *
import warnings
warnings.filterwarnings("ignore")

def engine(parameters, net, num_iter, LR, img_np, img_noisy_np):
  global i, out_avg, psrn_noisy_last, last_net, net_input
  optimizer = torch.optim.Adam(parameters, lr=LR)
  
  bar = tqdm(np.arange(num_iter), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
  for j in bar:
      optimizer.zero_grad()
      if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

      out = net(net_input)
      # Smoothing
      if out_avg is None:
          out_avg = out.detach()
      else:
          out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
              
      total_loss = mse(out, img_noisy_torch)
      total_loss.backward()
      psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
      psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
      psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

      content = f'Iteration: {j} , Loss: {total_loss.item():.3f} , PSNR_noisy: {psrn_noisy:.3f} , PSRN_gt:{psrn_gt:.3f}            '
      bar.set_description(content)
      if  PLOT and j % show_every == 0:
          out_np = torch_to_np(out)
          print('Iteration : ', j)
          plot_image_grid([np.clip(out_np, 0, 1), 
                            np.clip(torch_to_np(out_avg), 0, 1)],factor=12, nrow=8, labels = ['Output', 'Average Output'])
          
      # Backtracking
      if j % show_every:
          if psrn_noisy - psrn_noisy_last < -5: 
              print('Falling back to previous checkpoint.')

              for new_param, net_param in zip(last_net, net.parameters()):
                  net_param.data.copy_(new_param.cuda())

              return total_loss*0
          else:
              last_net = [x.detach().cpu() for x in net.parameters()]
              psrn_noisy_last = psrn_noisy
              
      optimizer.step()
