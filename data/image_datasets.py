import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import base64
import json
import argparse


def tupleize_data(images):
    temp = images
    tup = []

    for i in range(len(temp)):
        lst = temp[i]
        # print(lst)
        
        for idx, item in enumerate(lst):
            # print(str(item))
            # print(str(item)[-10:])
            if 'orig' in item.split('/')[-1]:
                img = images[i].pop(idx)
               
        for item in images[i]:
            tup.append((img, item))
    
    return tup

class PendulumPairedData(Dataset):
    def __init__(self, root, dataset="train"):
        self.root = root + "/" + dataset

        imgs = os.listdir(self.root)
        org_imgs = [] 
        for img in imgs:
            org_imgs.append(f'{img}.png')
        self.org_imgs = org_imgs

        imgs = [os.path.join(self.root, k) for k in imgs] # full path

        final_img = []
        for img in imgs:
            pairs = {}
            ims = os.listdir(img)
            ims = [os.path.join(img, k) for k in ims]
            
            keyword = 'orig'
            
            # get rid of orig
            ims = [s for s in ims if keyword not in s.split('/')[-1]]
            
            sorted_ims = sorted(ims, key=lambda x: x[-5])
            
            pairs[str(img)] = sorted_ims
            
            final_img.append(pairs)

        # self.data = tupleize_data(final_img)
        self.data = final_img # list of dictionaries corresponding to a folder

        self.transforms = transforms.Compose([transforms.ToTensor()])

    # discretize the labels
    def discretize_values(self, data, bins_dict):
        categorized_data = []
        for row in data:
            discrete_row = []
            for i in range(4):
                bins, labels = bins_dict[i]

                label_idx = np.digitize(row[i], bins, right=False) - 1  # Get category index
                label_idx = max(0, min(label_idx, len(labels) - 1))  # Ensure valid index
                discrete_row.append(labels[label_idx])
            categorized_data.append(discrete_row)
        return np.array(categorized_data)

    # Map numerical ranges to discretized categories
    def text_mapping(self, labels):
        # Define bins and labels for each column
        
        bins_dict = {
        0: ([-44, -6, 6, 44], ['left', 'center', 'right']),  # Bins for first variable
        1: ([60, 95, 105, 150], ['right', 'center', 'left']),  # Bins for second variable
        2: ([3, 6, 8, 12], ['short', 'medium', 'long']),  # Bins for third variable
        3: ([3, 7, 10, 16], ['left', 'center', 'right'])  # Bins for fourth variable
        }

        discretized_labels = self.discretize_values(labels, bins_dict)#.astype(object)
        df_discretized = pd.DataFrame(discretized_labels, columns=['angle', 'light_pos', 'shadow_len', 'shadow_pos'])

        return df_discretized

    # base-64 conversion
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def __getitem__(self, idx):

        orig_image = list(self.data[idx].keys())[0] # original image name
        orig_image_path = orig_image + '/' + orig_image.split('/')[-1] + '.png'
        pairs = self.data[idx][orig_image] # list of pairs sorted according to intervention target
    
        u_x = list(map(int, orig_image_path.split('/')[-1].strip('.png').split('_')[1:5]))
        
        # paired values
        u_y = [list(map(int, k.split('/')[-1].strip('.png').split('_')[1:5])) for k in pairs]

        # intervention target
        I = [list(map(int, k.split('/')[-1].strip('.png').split('_')[5])) for k in pairs]
        
        u = np.stack(u_x)
        u = np.expand_dims(u, axis=0)
        u_text = self.text_mapping(u).values.tolist()

        paired_values_discrete = []
        
        for i in range(len(u_y)):
            u_int = np.stack(u_y[i])
            u_int = np.expand_dims(u_int, axis=0)
            u_int_text = self.text_mapping(u_int).values.tolist()
            paired_values_discrete.append(u_int_text)

        return {
            "x_before_path": orig_image_path,
            "cf_paths": pairs,
            "int_target": I,
            "original_labels": u_text,
            "cf_labels": paired_values_discrete
        }
    
    def __len__(self):
        return len(self.data)
    
    
    
    
    
class FlowPairedData(Dataset):
    def __init__(self, root, dataset="train"):
        self.root = root + "/" + dataset

        imgs = os.listdir(self.root)
        org_imgs = [] 
        for img in imgs:
            org_imgs.append(f'{img}.png')
        self.org_imgs = org_imgs

        imgs = [os.path.join(self.root, k) for k in imgs] # full path

        final_img = []
        for img in imgs:
            pairs = {}
            ims = os.listdir(img)
            ims = [os.path.join(img, k) for k in ims]
            
            keyword = 'orig'
            
            # get rid of orig
            ims = [s for s in ims if keyword not in s.split('/')[-1]]
            
            sorted_ims = sorted(ims, key=lambda x: x[-5])
            
            pairs[str(img)] = sorted_ims
            
            final_img.append(pairs)

        # self.data = tupleize_data(final_img)
        self.data = final_img # list of dictionaries corresponding to a folder

        self.transforms = transforms.Compose([transforms.ToTensor()])

    # discretize the labels
    def discretize_values(self, data, bins_dict):
        categorized_data = []
        for row in data:
            discrete_row = []
            for i in range(4):
                bins, labels = bins_dict[i]

                label_idx = np.digitize(row[i], bins, right=False) - 1  # Get category index
                label_idx = max(0, min(label_idx, len(labels) - 1))  # Ensure valid index
                discrete_row.append(labels[label_idx])
            categorized_data.append(discrete_row)
        return np.array(categorized_data)
    
    # def __init__(self, root, dataset="train"):
    #     root = root + "/" + dataset

    #     imgs = os.listdir(root)

    #     imgs = [os.path.join(root, k) for k in imgs]

    #     final_img = []
    #     for img in imgs:
    #         ims = os.listdir(img)
    #         ims = [os.path.join(img, k) for k in ims]
    #         final_img.append(ims)

    #     self.data = tupleize_data(final_img)
        

    #     # Labels of pre and post intervention samples
    #     self.u_x = [list(map(int, k[0].split('/')[-1].strip('.png').split('_')[1:5])) for k in self.data]

    #     self.u_y = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[1:5])) for k in self.data]

    #     self.I = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[5])) for k in self.data]
    #     self.I_one_hot = np.eye(4)[self.I].astype('float32')

    #     self.transforms = transforms.Compose([transforms.ToTensor()])

    # # discretize the labels
    # def discretize_values(self, data, bins_dict):
    #     categorized_data = []
    #     for row in data:
    #         discrete_row = []
    #         for i in range(4):
    #             bins, labels = bins_dict[i]

    #             label_idx = np.digitize(row[i], bins, right=False) - 1  # Get category index
    #             label_idx = max(0, min(label_idx, len(labels) - 1))  # Ensure valid index
    #             discrete_row.append(labels[label_idx])
    #         categorized_data.append(discrete_row)
    #     return np.array(categorized_data)

    # Map numerical ranges to discretized categories
    def text_mapping(self, labels):
        # Define bins and labels for each column
        bins_dict = {
        0: ([5, 17, 23, 36], ['small', 'medium', 'large']),  # Bins for first variable (ball size)
        1: ([6, 9, 12, 15], ['bottom', 'middle', 'top']),  # Bins for second variable (hole position)
        2: ([1, 2, 3, 4], ['low', 'medium', 'high']),  # Bins for third variable (water level)
        3: ([17, 37, 47, 67], ['left', 'middle', 'right'])  # Bins for fourth variable (water flow)
        }

        discretized_labels = self.discretize_values(labels, bins_dict)#.astype(object)
        df_discretized = pd.DataFrame(discretized_labels, columns=['ball', 'hole', 'water_level', 'water_flow'])

        return df_discretized

    # base-64 conversion
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def __getitem__(self, idx):

        orig_image = list(self.data[idx].keys())[0] # original image name
        orig_image_path = orig_image + '/' + orig_image.split('/')[-1] + '.png'
        pairs = self.data[idx][orig_image] # list of pairs sorted according to intervention target
    
        u_x = list(map(int, orig_image_path.split('/')[-1].strip('.png').split('_')[1:5]))
        
        # paired values
        u_y = [list(map(int, k.split('/')[-1].strip('.png').split('_')[1:5])) for k in pairs]

        # intervention target
        I = [list(map(int, k.split('/')[-1].strip('.png').split('_')[5])) for k in pairs]
        
        u = np.stack(u_x)
        u = np.expand_dims(u, axis=0)
        u_text = self.text_mapping(u).values.tolist()

        paired_values_discrete = []
        
        for i in range(len(u_y)):
            u_int = np.stack(u_y[i])
            u_int = np.expand_dims(u_int, axis=0)
            u_int_text = self.text_mapping(u_int).values.tolist()
            paired_values_discrete.append(u_int_text)

        return {
            "x_before_path": orig_image_path,
            "cf_paths": pairs,
            "int_target": I,
            "original_labels": u_text,
            "cf_labels": paired_values_discrete
        }
    
    def __len__(self):
        return len(self.data)
    # def __getitem__(self, idx):
    #     # get paths
    #     img_path_1 = self.data[idx][0]
    #     img_path_2 = self.data[idx][1]
        
    #     # original image and text
    #     # x = np.asarray(PILImage.open(img_path_1).convert("RGB"))#.convert("RGB"))
    #     u = np.stack(self.u_x[idx])
    #     u = np.expand_dims(u, axis=0)
    #     u_text = self.text_mapping(u).values.tolist()

    #     # counterfactual image and text
    #     # x_int = np.asarray(PILImage.open(img_path_2).convert("RGB"))#.convert("RGB"))
    #     u_int = np.stack(self.u_y[idx])
    #     u_int = np.expand_dims(u_int, axis=0)
    #     u_int_text = self.text_mapping(u_int).values.tolist()

    #     # intervention target
    #     # target = torch.from_numpy(np.asarray(self.I_one_hot[idx]))
    #     target = torch.from_numpy(np.asarray(self.I[idx]))

    #     # return dictionary
    #     return {
    #         "x_before_path": img_path_1,
    #         "x_after_path": img_path_2,
    #         "int_target": target,
    #         "original_labels": u_text,
    #         "cf_labels": u_int_text
    #     }
    
    # def __len__(self):
    #     return len(self.data)
    






class CircuitPairedData(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)

        imgs = [os.path.join(root, k) for k in imgs]

        final_img = []
        for img in imgs:
            ims = os.listdir(img)
            ims = [os.path.join(img, k) for k in sorted(ims)]
            final_img.append(ims)
        
        self.data = final_img

        # Labels of pre and post intervention samples
        self.u_x = [list(map(float, k[0].split('/')[-1].strip('.png').split('_')[1:5])) for k in final_img]

        self.u_y = [list(map(float, k[1].split('/')[-1].strip('.png').split('_')[1:5])) for k in final_img]

        self.I = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[5][1])) for k in final_img]

        self.transforms = transforms.Compose([transforms.ToTensor()])

    # discretize the labels
    def discretize_values(self, data, bins_dict):
        categorized_data = []
        for row in data:
            discrete_row = []
            for i in range(4):
                bins, labels = bins_dict[i]

                label_idx = np.digitize(row[i], bins, right=False) - 1  # Get category index
                label_idx = max(0, min(label_idx, len(labels) - 1))  # Ensure valid index
                discrete_row.append(labels[label_idx])
            categorized_data.append(discrete_row)
        return np.array(categorized_data)

    # Map numerical ranges to discretized categories
    def text_mapping(self, labels):
        # Define bins and labels for each column
        bins_dict = {
        0: ([0.0, 0.45, 1.0], ['off', 'on']),  # Bins for second variable
        1: ([0.0, 0.45, 1.0], ['off', 'on']),  # Bins for third variable
        2: ([0.0, 0.45, 1.0], ['off', 'on']),  # Bins for fourth variable
        3: ([0.0, 0.2, 0.33, 0.43, 0.58, 0.74, 0.9, 1.0], ['not touching any light', 'touching green light', 'not touching any light', 'touching red light', 'not touching any light', 'touching blue light', 'not touching any light']),  # Bins for first variable
        }

        discretized_labels = self.discretize_values(labels, bins_dict)#.astype(object)
        df_discretized = pd.DataFrame(discretized_labels, columns=['red', 'green', 'blue', 'arm'])

        return df_discretized

    # base-64 conversion
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def __getitem__(self, idx):
        # get paths
        img_path_1 = self.data[idx][0]
        img_path_2 = self.data[idx][1]
        
        # base-64 images
        # x_pre = self.encode_image(img_path_1)
        # x_post = self.encode_image(img_path_2)

        # original image and text
        # x = np.asarray(PILImage.open(img_path_1).convert("RGB"))#.convert("RGB"))
        u = np.stack(self.u_x[idx])
        u = np.expand_dims(u, axis=0)
        u_text = self.text_mapping(u).values.tolist()
        
        # counterfactual image and text
        # x_int = np.asarray(PILImage.open(img_path_2).convert("RGB"))#.convert("RGB"))
        u_int = np.stack(self.u_y[idx])
        u_int = np.expand_dims(u_int, axis=0)
        u_int_text = self.text_mapping(u_int).values.tolist()

        # intervention target
        target = torch.from_numpy(np.asarray(self.I[idx]))

        # process images
        # if self.transforms:
        #     x = self.transforms(x)
        #     x_int = self.transforms(x_int)
        # else:
        #     x = np.from_numpy(np.asarray(x).reshape(96, 96, 4))
        #     x_int = np.from_numpy(np.asarray(x_int).reshape(96, 96, 4))

        # return dictionary
        # return {
        #     "x_before_path": img_path_1,
        #     "x_after_path": img_path_2,
        #     "int_target": target
        # }
        
        return {
            "x_before_path": img_path_1,
            "cf_paths": img_path_2,
            "int_target": target,
            "original_labels": u_text,
            "cf_labels": u_int_text
        }
    
    def __len__(self):
        return len(self.data)
    






class SyntheticPaired(Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset

        imgs = os.listdir(root)

        imgs = [os.path.join(root, k) for k in imgs]

        final_img = []
        for img in imgs:
            ims = os.listdir(img)
            ims = [os.path.join(img, k) for k in ims]
            final_img.append(ims)

        self.data = tupleize_data(final_img)
        

        # Labels of pre and post intervention samples
        self.u_x = [list(map(int, k[0].split('/')[-1].strip('.png').split('_')[1:5])) for k in self.data]

        # self.u_x = [list(map(int, k[0].split('/')[6].strip('.png').split('_')[1:6])) for k in self.data]
        self.u_y = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[1:5])) for k in self.data]

        # Intervention Target
        # self.I = [list(map(int, k[1][-7:-4].split('_')[1:])) for k in self.data]

        self.I = [list(map(int, k[1].split('/')[-1].strip('.png').split('_')[5])) for k in self.data]

        self.I_one_hot = np.eye(4)[self.I].astype('float32')

        self.transforms = transforms.Compose([transforms.ToTensor()])


    def discretize_values(self, data, bins_dict):
        categorized_data = []
        for row in data:
            discrete_row = []
            for i in range(4):
                bins, labels = bins_dict[i]

                label_idx = np.digitize(row[i], bins, right=False) - 1  # Get category index
                label_idx = max(0, min(label_idx, len(labels) - 1))  # Ensure valid index
                discrete_row.append(labels[label_idx])
            categorized_data.append(discrete_row)
        return np.array(categorized_data)


    def text_mapping(self, labels):
        # Define bins and labels for each column
        bins_dict = {
        0: ([-44, -27, -6, 6, 26, 44], ['Far Left', 'Left', 'Center', 'Right', 'Far Right']),  # Bins for first variable
        1: ([60, 90, 120, 150], ['Right', 'Center', 'Left']),  # Bins for second variable
        2: ([3, 5, 7, 10], ['Short', 'Medium', 'Long']),  # Bins for third variable
        3: ([3, 6, 9, 11, 13, 16], ['Far Left', 'Left', 'Center', 'Right', 'Far Right'])  # Bins for fourth variable
        }

        discretized_labels = self.discretize_values(labels, bins_dict)#.astype(object)
        # print(discretized_labels)
        # exit(0)

        df_discretized = pd.DataFrame(discretized_labels, columns=['angle', 'light_pos', 'shadow_len', 'shadow_pos'])

        return df_discretized

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def __getitem__(self, idx):
        img_path_1 = self.data[idx][0]
        img_path_2 = self.data[idx][1]
        
        x_pre = self.encode_image(img_path_1)
        x_post = self.encode_image(img_path_2)

        # x = np.asarray(PILImage.open(img_path_1))
        u = np.stack(self.u_x[idx])
        u = np.expand_dims(u, axis=0)
        u_text = self.text_mapping(u)
        
        # x_int = np.asarray(PILImage.open(img_path_2))
        u_int = np.stack(self.u_y[idx])
        u_int = np.expand_dims(u_int, axis=0)
        u_int_text = self.text_mapping(u_int)

        target = torch.from_numpy(np.asarray(self.I_one_hot[idx]))

        # if self.transforms:
        #     x = self.transforms(x)
        #     x_int = self.transforms(x_int)
        # else:
        #     x = np.from_numpy(np.asarray(x).reshape(96, 96, 4))
        #     x_int = np.from_numpy(np.asarray(x_int).reshape(96, 96, 4))
        # return [img_path_1, u_text.values.tolist(), img_path_2, u_int_text.values.tolist(), target]

        return {
            "x_before": x_pre,
            "before_text": u_text.values.tolist(),
            "x_after": x_post,
            "after_text": u_int_text.values.tolist(),
            "target": target
        }
    
    def __len__(self):
        return len(self.data)