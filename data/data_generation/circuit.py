import io
import os
import math
import random
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset


def save_images():
    data = np.load(f'./circuit/train.npz')

    img_before = data['imgs'][:, 0]
    img_after = data['imgs'][:, 1]


    intervention_target = data['intervention_labels']

    img_labels = data['original_latents'][:, 0, :]
    post_labels = data['original_latents'][:, 1, :]
    
    
    indices_11 = np.argwhere(((img_labels[:, 0] == post_labels[:, 0]) | (np.abs(post_labels[:, 0] - img_labels[:, 0] > 0.5)) ) & 
                             ((img_labels[:, 1] == post_labels[:, 1]) | (np.abs(post_labels[:, 1] - img_labels[:, 1] > 0.5)) ) & 
                             ((img_labels[:, 2] == post_labels[:, 2]) | (np.abs(post_labels[:, 2] - img_labels[:, 2] > 0.5)) ) &
                             ((img_labels[:, 0] != post_labels[:, 0]) | (img_labels[:, 1] != post_labels[:, 1]) | (img_labels[:, 2] != post_labels[:, 2])))
    

    
    img_labels_1 = img_labels[((img_labels[:, 0] == post_labels[:, 0]) | (np.abs(post_labels[:, 0] - img_labels[:, 0] > 0.5)) ) & 
                             ((img_labels[:, 1] == post_labels[:, 1]) | (np.abs(post_labels[:, 1] - img_labels[:, 1] > 0.5)) ) & 
                             ((img_labels[:, 2] == post_labels[:, 2]) | (np.abs(post_labels[:, 2] - img_labels[:, 2] > 0.5)) ) &
                             ((img_labels[:, 0] != post_labels[:, 0]) | (img_labels[:, 1] != post_labels[:, 1]) | (img_labels[:, 2] != post_labels[:, 2]))]
    
    post_labels_1 = post_labels[((img_labels[:, 0] == post_labels[:, 0]) | (np.abs(post_labels[:, 0] - img_labels[:, 0] > 0.5)) ) & 
                             ((img_labels[:, 1] == post_labels[:, 1]) | (np.abs(post_labels[:, 1] - img_labels[:, 1] > 0.5)) ) & 
                             ((img_labels[:, 2] == post_labels[:, 2]) | (np.abs(post_labels[:, 2] - img_labels[:, 2] > 0.5)) ) &
                             ((img_labels[:, 0] != post_labels[:, 0]) | (img_labels[:, 1] != post_labels[:, 1]) | (img_labels[:, 2] != post_labels[:, 2]))]


   
    img_labels = img_labels_1
    post_labels = post_labels_1
    indices = indices_11
    
    # indices = np.concatenate((indices_11, indices_12, indices_13))
    intervention_target = intervention_target[indices.tolist()]

    indices = indices[intervention_target[:, 0] != 0]
    img_labels = img_labels[intervention_target[:, 0] != 0]
    post_labels = post_labels[intervention_target[:, 0] != 0]
    intervention_target = intervention_target[intervention_target[:, 0] != 0]

    temp = np.concatenate((img_before.reshape(-1, 1), img_after.reshape(-1, 1)), axis=1)

    filtered_images = temp[indices.tolist()]
    filtered_images = filtered_images.squeeze(1)


    for i in range(filtered_images.shape[0]):
        im_b = Image.open(io.BytesIO(filtered_images[i][0]))
        im_a = Image.open(io.BytesIO(filtered_images[i][1]))
        
        if not os.path.exists(f'./circuit/circuit_paired/a_{str(img_labels[i][0])}_{str(img_labels[i][1])}_{str(img_labels[i][2])}_{str(img_labels[i][3])}_orig'):
            os.makedirs(f'./circuit/circuit_paired/a_{str(img_labels[i][0])}_{str(img_labels[i][1])}_{str(img_labels[i][2])}_{str(img_labels[i][3])}_orig')

        folder = f'./circuit/circuit_paired/a_{str(img_labels[i][0])}_{str(img_labels[i][1])}_{str(img_labels[i][2])}_{str(img_labels[i][3])}_orig'
        im_b.save(f'./{folder}/a_{str(img_labels[i][0])}_{str(img_labels[i][1])}_{str(img_labels[i][2])}_{str(img_labels[i][3])}_orig.png')
        im_a.save(f'./{folder}/i_{str(post_labels[i][0])}_{str(post_labels[i][1])}_{str(post_labels[i][2])}_{str(post_labels[i][3])}_{intervention_target[i]}.png')



class CausalCircuit(Dataset):
    def __init__(self, root, dataset="train"):
        
        self.imgs = []
        self.labels = []
        
        data = np.load(f'{root}/train.npz')

        img_before = data['imgs'][:, 0]
        img_after = data['imgs'][:, 1]

        intervention_target = data['intervention_labels']

    def __getitem__(self, idx):
        #print(idx)
        data = self.imgs[idx]
        
        return data#, label.float()

    def __len__(self):
        return len(self.imgs)

save_images()