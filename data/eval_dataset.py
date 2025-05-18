import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
# import random
import base64
import json
import argparse
from image_datasets import PendulumPairedData, FlowPairedData, CircuitPairedData

    
data = "flow"
task = "discovery_interleaved"
def main():
    if data == "pendulum":
        mapper = {'pendulum angle': '0', 'light position': '1', 'shadow length': '2', 'shadow position': '3'}
        reverse_mapper = {0:'pendulum angle', 1: 'light position', 2: 'shadow length', 3: 'shadow position'}

        dataset = PendulumPairedData('../../datasets/new_data/pendulum')
        
        train_ratio = 0.4
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        jsondata = []
        jsondata_train = []
        jsondata_test = []
        variables = ["pendulum angle", "light position", "shadow length", "shadow position"]
        
        # TRAINING (SUPPORT) SET
        for idx, batch in enumerate(train_dataloader):

            x_before, x_after, target, original_labels, cf_labels = batch['x_before_path'], batch['cf_paths'], batch["int_target"], batch["original_labels"], batch["cf_labels"]

            # x_before = batch['x_before_path']
            
            if task == 'counterfactual' or task == 'intervention' or task == 'discovery_interleaved':
                for i in range(4):
                    interv = target[i][0].item()
                    if task == "counterfactual":
                        
                        jsondata_train.append({
                                "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                                "image": [x_before[0]], 
                                # "question": "From the first to the second image, which variable is the primary variable that changes?",
                                "question": f"In the given image, the values of the variables are given as\npendulum angle: {original_labels[0][0][0]}\nlight position: {original_labels[0][1][0]}\nshadow position: {original_labels[0][3][0]}\nshadow length: {original_labels[0][2][0]}\nIf the {reverse_mapper[interv]} is changed from {original_labels[0][interv][0]} to {cf_labels[i][0][interv][0]}, what are the final values of all variables? Answer concisely with the specific values that each variable will take.",
                                "answer": f"\npendulum angle: {str(cf_labels[i][0][0][0])}\nlight position: {str(cf_labels[i][0][1][0])}\nshadow position: {str(cf_labels[i][0][3][0])}\nshadow length: {str(cf_labels[i][0][2][0])}",
                                "intervention":  reverse_mapper[interv],
                                "angle_orig": str(original_labels[0][0][0]),
                                "light_orig": str(original_labels[0][1][0]),
                                "shadow_pos_orig": str(original_labels[0][3][0]),
                                "shadow_len_orig": str(original_labels[0][2][0]),
                                "angle_cf": str(cf_labels[i][0][0][0]),
                                "light_cf": str(cf_labels[i][0][1][0]),
                                "shadow_pos_cf": str(cf_labels[i][0][3][0]),
                                "shadow_len_cf": str(cf_labels[i][0][2][0])
                            })
                    elif task == "intervention":
                        jsondata_train.append({
                            "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0], x_after[i][0]], 
                            # "question": "From the first to the second image, which variable is the primary variable that changes?",
                            "question": "From the first to the second image, which variable changes first?",
                            "answer":  reverse_mapper[interv]
                        })
                    elif task == "discovery_interleaved":
                        for j in range(len(variables)):
                            new_var = variables.copy()
                            new_var.pop(j)
                            for k in range(len(variables) - 1):
                                question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                                jsondata_train.append({
                                    "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                                    "image": [x_before[0], x_after[i][0]], 
                                    "question": question,
                                    "answer":  "Yes" if question == "Does pendulum angle directly cause shadow length to change?" or 
                                    question == "Does light position directly cause shadow length to change?" or 
                                    question == "Does pendulum angle directly cause shadow position to change?" or 
                                    question == "Does light position directly cause shadow position to change?" else "No"
                                })
            else:
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_train.append({
                            "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does pendulum angle directly cause shadow length to change?" or 
                            question == "Does light position directly cause shadow length to change?" or 
                            question == "Does pendulum angle directly cause shadow position to change?" or 
                            question == "Does light position directly cause shadow position to change?" else "No"
                        })
        
        # TESTING (QUERY) SET
        for idx, batch in enumerate(test_dataloader):

            x_before, x_after, target, original_labels, cf_labels = batch['x_before_path'], batch['cf_paths'], batch["int_target"], batch["original_labels"], batch["cf_labels"]
            
            if task == 'counterfactual' or task == 'intervention' or task == 'discovery_interleaved':
                for i in range(4):
                    interv = target[i][0].item()
                    if task == "counterfactual":
                        jsondata_test.append({
                                "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                                "image": [x_before[0]], 
                                # "question": "From the first to the second image, which variable is the primary variable that changes?",
                                "question": f"In the given image, the values of the variables are given as\npendulum angle: {original_labels[0][0][0]}\nlight position: {original_labels[0][1][0]}\nshadow position: {original_labels[0][3][0]}\nshadow length: {original_labels[0][2][0]}\nIf the {reverse_mapper[interv]} is changed from {original_labels[0][interv][0]} to {cf_labels[i][0][interv][0]}, what are the final values of all variables? Answer concisely with the specific values that each variable will take.",
                                "answer": f"\npendulum angle: {str(cf_labels[i][0][0][0])}\nlight position: {str(cf_labels[i][0][1][0])}\nshadow position: {str(cf_labels[i][0][3][0])}\nshadow length: {str(cf_labels[i][0][2][0])}",
                                "intervention":  reverse_mapper[interv],
                                "angle_orig": str(original_labels[0][0][0]),
                                "light_orig": str(original_labels[0][1][0]),
                                "shadow_pos_orig": str(original_labels[0][3][0]),
                                "shadow_len_orig": str(original_labels[0][2][0]),
                                "angle_cf": str(cf_labels[i][0][0][0]),
                                "light_cf": str(cf_labels[i][0][1][0]),
                                "shadow_pos_cf": str(cf_labels[i][0][3][0]),
                                "shadow_len_cf": str(cf_labels[i][0][2][0])
                            })
                    elif task == "intervention":
                        jsondata_test.append({
                            "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0], x_after[i][0]], 
                            # "question": "From the first to the second image, which variable is the primary variable that changes?",
                            "question": "From the first to the second image, which variable changes first?",
                            "answer":  reverse_mapper[interv]
                        })
                    elif task == "discovery_interleaved":
                        for j in range(len(variables)):
                            new_var = variables.copy()
                            new_var.pop(j)
                            for k in range(len(variables) - 1):
                                question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                                jsondata_test.append({
                                    "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}_{i}", 
                                    "image": [x_before[0], x_after[i][0]], 
                                    "question": question,
                                    "answer":  "Yes" if question == "Does pendulum angle directly cause shadow length to change?" or 
                                    question == "Does light position directly cause shadow length to change?" or 
                                    question == "Does pendulum angle directly cause shadow position to change?" or 
                                    question == "Does light position directly cause shadow position to change?" else "No"
                                })
            else:
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_test.append({
                            "id": f"pendulum_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does pendulum angle directly cause shadow length to change?" or 
                            question == "Does light position directly cause shadow length to change?" or 
                            question == "Does pendulum angle directly cause shadow position to change?" or 
                            question == "Does light position directly cause shadow position to change?" else "No"
                        })
        
        with open(f"../VL-ICL/new_data/{task}/{data}/query.json", "w") as f:
            # seen = set()
            # unique_data = []
            # comparison_keys = ['id', 'question'] 
            # for entry in jsondata_test:
            #     key = tuple(entry.get(k) for k in comparison_keys)  # Use tuple for hashable key

            #     if key not in seen:
            #         seen.add(key)
            #         unique_data.append(entry)
                    
            # jsondata_test = unique_data
            
            json.dump(jsondata_test, f, indent=4, ensure_ascii=False)    
                    
        with open(f"../VL-ICL/new_data/{task}/{data}/support.json", "w") as f:
            # seen = set()
            # unique_data = []
            # comparison_keys = ['id', 'question'] 
            
            # for entry in jsondata_train:
            #     key = tuple(entry.get(k) for k in comparison_keys)  # Use tuple for hashable key

            #     if key not in seen:
            #         seen.add(key)
            #         unique_data.append(entry)
                    
            # jsondata_train = unique_data
            
            json.dump(jsondata_train, f, indent=4, ensure_ascii=False) 

    elif data == "flow":

        mapper = {'ball size': '0', 'hole position': '1', 'water level': '2', 'water flow': '3'}
        reverse_mapper = {0:'ball size', 1: 'hole position', 2: 'water level', 3: 'water flow'} 

        dataset = FlowPairedData('../../datasets/new_data/flow')
        
        train_ratio = 0.4
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        jsondata = []
        jsondata_train = []
        jsondata_test = []
        variables = ["ball size", "hole position", "water level", "water flow"]
        
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # jsondata = []
        for idx, batch in enumerate(train_dataloader):
            
            x_before, x_after, target, original_labels, cf_labels = batch['x_before_path'], batch['cf_paths'], batch["int_target"], batch["original_labels"], batch["cf_labels"]


            if task == 'counterfactual' or task == 'intervention' or task == 'discovery_interleaved':
                for i in range(3):
                    interv = target[i][0].item()
                    if task == "counterfactual":
                        jsondata_train.append({
                                "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                                "image": [x_before[0]], 
                                "question": f"In the given image, the values of the variables are given as\nball size: {original_labels[0][0][0]}\nhole position: {original_labels[0][1][0]}\nwater level: {original_labels[0][2][0]}\nwater flow: {original_labels[0][3][0]}\nIf the {reverse_mapper[interv]} is changed from {original_labels[0][interv][0]} to {cf_labels[i][0][interv][0]}, what are the final values of all variables? Answer concisely with the specific values that each variable will take.",
                                "answer": f"\nball size: {str(cf_labels[i][0][0][0])}\nhole position: {str(cf_labels[i][0][1][0])}\nwater level: {str(cf_labels[i][0][2][0])}\nwater flow: {str(cf_labels[i][0][3][0])}",
                                "intervention":  reverse_mapper[interv],
                                "ball_orig": str(original_labels[0][0][0]),
                                "hole_orig": str(original_labels[0][1][0]),
                                "water_level_orig": str(original_labels[0][2][0]),
                                "water_flow_orig": str(original_labels[0][3][0]),
                                "ball_cf": str(cf_labels[i][0][0][0]),
                                "hole_cf": str(cf_labels[i][0][1][0]),
                                "water_level_cf": str(cf_labels[i][0][2][0]),
                                "water_flow_cf": str(cf_labels[i][0][3][0])
                            })
                    elif task == "intervention":
                        jsondata_train.append({
                            "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0], x_after[i][0]], 
                            # "question": "From the first to the second image, which variable is the primary variable that changes?",
                            "question": "From the first to the second image, which variable changes first?",
                            "answer":  reverse_mapper[interv]
                        })
                    elif task == "discovery_interleaved":
                        for j in range(len(variables)):
                            new_var = variables.copy()
                            new_var.pop(j)
                            for k in range(len(variables) - 1):
                                question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                                jsondata_train.append({
                                    "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                                    "image": [x_before[0], x_after[i][0]], 
                                    "question": question,
                                    "answer":  "Yes" if question == "Does ball size directly cause water level to change?" or 
                                    question == "Does water level directly cause water flow to change?" or 
                                    question == "Does hole position directly cause water flow to change?" else "No"
                                })
            else:
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_train.append({
                            "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does ball size directly cause water level to change?" or 
                            question == "Does water level directly cause water flow to change?" or 
                            question == "Does hole position directly cause water flow to change?" else "No"
                        })
                
                
        
        
        
        
        for idx, batch in enumerate(test_dataloader):
            
            x_before, x_after, target, original_labels, cf_labels = batch['x_before_path'], batch['cf_paths'], batch["int_target"], batch["original_labels"], batch["cf_labels"]
            # x_before = batch['x_before_path']
            # x_after = batch['x_after_path']
            # target = batch["int_target"]
            # original_labels = batch["original_labels"]
            # cf_labels = batch["cf_labels"]

            if task == 'counterfactual' or task == 'intervention' or task == 'discovery_interleaved':
                for i in range(3):
                    interv = target[i][0].item()
                    if task == "counterfactual":
                        jsondata_test.append({
                                "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                                "image": [x_before[0]], 
                                "question": f"In the given image, the values of the variables are given as\nball size: {original_labels[0][0][0]}\nhole position: {original_labels[0][1][0]}\nwater level: {original_labels[0][2][0]}\nwater flow: {original_labels[0][3][0]}\nIf the {reverse_mapper[interv]} is changed from {original_labels[0][interv][0]} to {cf_labels[i][0][interv][0]}, what are the final values of all variables? Answer concisely with the specific values that each variable will take.",
                                "answer": f"\nball size: {str(cf_labels[i][0][0][0])}\nhole position: {str(cf_labels[i][0][1][0])}\nwater level: {str(cf_labels[i][0][2][0])}\nwater flow: {str(cf_labels[i][0][3][0])}",
                                "intervention":  reverse_mapper[interv],
                                "ball_orig": str(original_labels[0][0][0]),
                                "hole_orig": str(original_labels[0][1][0]),
                                "water_level_orig": str(original_labels[0][2][0]),
                                "water_flow_orig": str(original_labels[0][3][0]),
                                "ball_cf": str(cf_labels[i][0][0][0]),
                                "hole_cf": str(cf_labels[i][0][1][0]),
                                "water_level_cf": str(cf_labels[i][0][2][0]),
                                "water_flow_cf": str(cf_labels[i][0][3][0])
                            })
                    elif task == "intervention":
                        jsondata_test.append({
                            "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0], x_after[i][0]], 
                            # "question": "From the first to the second image, which variable is the primary variable that changes?",
                            "question": "From the first to the second image, which variable changes first?",
                            "answer":  reverse_mapper[interv]
                        })
                    elif task == "discovery_interleaved":
                        for j in range(len(variables)):
                            new_var = variables.copy()
                            new_var.pop(j)
                            for k in range(len(variables) - 1):
                                question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                                jsondata_test.append({
                                    "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}_{i}", 
                                    "image": [x_before[0], x_after[i][0]], 
                                    "question": question,
                                    "answer":  "Yes" if question == "Does ball size directly cause water level to change?" or 
                                    question == "Does water level directly cause water flow to change?" or 
                                    question == "Does hole position directly cause water flow to change?" else "No"
                                })
            else:
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_test.append({
                            "id": f"flow_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does ball size directly cause water level to change?" or 
                            question == "Does water level directly cause water flow to change?" or 
                            question == "Does hole position directly cause water flow to change?" else "No"
                        })
                
        with open(f"../VL-ICL/new_data/{task}/{data}/query.json", "w") as f:
            # seen = set()
            # unique_data = []
            # comparison_keys = ['id', 'question'] 
            # for entry in jsondata_test:
            #     key = tuple(entry.get(k) for k in comparison_keys)  # Use tuple for hashable key

            #     if key not in seen:
            #         seen.add(key)
            #         unique_data.append(entry)
                    
            # jsondata_test = unique_data
            json.dump(jsondata_test, f, indent=4, ensure_ascii=False)
        
        with open(f"../VL-ICL/new_data/{task}/{data}/support.json", "w") as f:
            # seen = set()
            # unique_data = []
            # comparison_keys = ['id', 'question'] 
            
            # for entry in jsondata_train:
            #     key = tuple(entry.get(k) for k in comparison_keys)  # Use tuple for hashable key

            #     if key not in seen:
            #         seen.add(key)
            #         unique_data.append(entry)
                    
            # jsondata_train = unique_data
            json.dump(jsondata_train, f, indent=4, ensure_ascii=False)
    else:
        reverse_mapper = {1:'red light', 2: 'green light', 3: 'blue light', 4: 'robot arm'}
        dataset = CircuitPairedData('../../datasets/circuit/circuit_paired')
        
        train_ratio = 0.4
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        jsondata = []
        jsondata_train = []
        jsondata_test = []
        variables = ["robot arm position", "green light", "blue light", "red light"]
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # jsondata = []
        for idx, batch in enumerate(train_dataloader):
            
            x_before, x_after, target, original_labels, cf_labels = batch['x_before_path'], batch['cf_paths'], batch["int_target"], batch["original_labels"], batch["cf_labels"]
            
            # x_before = batch['x_before_path']
            # x_after = batch['cf_paths']
            # target = batch["int_target"]
            # print(cf_labels[0][3][0])
            interv = target.item() - 1
 
            if task == "counterfactual":
                jsondata_train.append({
                        "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                        "image": [x_before[0]], 
                        "question": f"In the given image, the values of the variables are given as\nred light: {original_labels[0][0][0]}\ngreen light: {original_labels[0][1][0]}\nblue light: {original_labels[0][2][0]}\nrobot arm: {original_labels[0][3][0]}\nIf the {reverse_mapper[interv + 1]} is changed from {original_labels[0][interv][0]} to {cf_labels[0][interv][0]}, what are the final values of all variables? Answer concisely with the specific values that each variable will take.",
                        "answer": f"\nred light: {str(cf_labels[0][0][0])}\ngreen light: {str(cf_labels[0][1][0])}\nblue light: {str(cf_labels[0][2][0])}\nrobot arm: {str(cf_labels[0][3][0])}",
                        "intervention":  reverse_mapper[interv + 1],
                        "red light original": str(original_labels[0][0][0]),
                        "green light original": str(original_labels[0][1][0]),
                        "blue light original": str(original_labels[0][2][0]),
                        "robot arm original": str(original_labels[0][3][0]),
                        "red light cf": str(cf_labels[0][0][0]),
                        "green light cf": str(cf_labels[0][1][0]),
                        "blue light cf": str(cf_labels[0][2][0]),
                        "robot arm cf": str(cf_labels[0][3][0])
                    })
            elif task == "intervention":
                jsondata_train.append({
                "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                "image": [x_before[0], x_after[0]], 
                # "question": "From the first to the second image, which variable is the primary variable that changes?",
                "question": "From the first to the second image, which variable changes first?",
                "answer":  reverse_mapper[target.item()]
            })
            #     jsondata_train.append({
            #     "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
            #     "image": [x_before[0], x_after[0]], 
            #     # "question": "From the first to the second image, which variable is the primary variable that changes?",
            #     "question": "From the first to the second image, which variable changes first?",
            #     "answer":  reverse_mapper[target.item()]
            # })
        
            elif task == "discovery_interleaved":
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_train.append({
                            "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0], x_after[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does robot arm position directly cause red light to change?" or 
                            question == "Does robot arm position directly cause red light to change?" or 
                            question == "Does robot arm position directly cause green light to change?" or
                            question == "Does robot arm position directly cause blue light to change?" or 
                            question == "Does blue light directly cause red light to change?" or
                            question == "Does green light directly cause red light to change?" else "No"
                        })
            else:
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_train.append({
                            "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does robot arm position directly cause red light to change?" or 
                                    question == "Does robot arm position directly cause red light to change?" or 
                                    question == "Does robot arm position directly cause green light to change?" or
                                    question == "Does robot arm position directly cause blue light to change?" or 
                                    question == "Does blue light directly cause red light to change?" or
                                    question == "Does green light directly cause red light to change?" else "No"
                        })
                
        
        
        for idx, batch in enumerate(test_dataloader):

            # x_before = batch['x_before_path']
            # x_after = batch['x_after_path']
            # target = batch["int_target"]

            x_before, x_after, target, original_labels, cf_labels = batch['x_before_path'], batch['cf_paths'], batch["int_target"], batch["original_labels"], batch["cf_labels"]
            interv = target.item() - 1
            
            if task == "counterfactual":
                jsondata_test.append({
                        "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                        "image": [x_before[0]], 
                        "question": f"In the given image, the values of the variables are given as\nred light: {original_labels[0][0][0]}\ngreen light: {original_labels[0][1][0]}\nblue light: {original_labels[0][2][0]}\nrobot arm: {original_labels[0][3][0]}\nIf the {reverse_mapper[interv + 1]} is changed from {original_labels[0][interv][0]} to {cf_labels[0][interv][0]}, what are the final values of all variables? Answer concisely with the specific values that each variable will take.",
                        "answer": f"\nred light: {str(cf_labels[0][0][0])}\ngreen light: {str(cf_labels[0][1][0])}\nblue light: {str(cf_labels[0][2][0])}\nrobot arm: {str(cf_labels[0][3][0])}",
                        "intervention":  reverse_mapper[interv + 1],
                        "red light original": str(original_labels[0][0][0]),
                        "green light original": str(original_labels[0][1][0]),
                        "blue light original": str(original_labels[0][2][0]),
                        "robot arm original": str(original_labels[0][3][0]),
                        "red light cf": str(cf_labels[0][0][0]),
                        "green light cf": str(cf_labels[0][1][0]),
                        "blue light cf": str(cf_labels[0][2][0]),
                        "robot arm cf": str(cf_labels[0][3][0])
                    })
            elif task == "intervention":
                jsondata_test.append({
                    "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                    "image": [x_before[0], x_after[0]], 
                    # "question": "From the first to the second image, which variable is the primary variable that changes?",
                    "question": "From the first to the second image, which variable changes first?",
                    "answer":  reverse_mapper[target.item()]
                })
            
            elif task == "discovery_interleaved":
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_test.append({
                            "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0], x_after[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does robot arm position directly cause red light to change?" or 
                            question == "Does robot arm position directly cause red light to change?" or 
                            question == "Does robot arm position directly cause green light to change?" or
                            question == "Does robot arm position directly cause blue light to change?" or 
                            question == "Does blue light directly cause red light to change?" or
                            question == "Does green light directly cause red light to change?" else "No"
                        })
            else:
                for j in range(len(variables)):
                    new_var = variables.copy()
                    new_var.pop(j)
                    for k in range(len(variables) - 1):
                        question = f"Does {variables[j]} directly cause {new_var[k]} to change?"
                        jsondata_test.append({
                            "id": f"circuit_{x_before[0].split('/')[-1].split('.png')[0]}", 
                            "image": [x_before[0]], 
                            "question": question,
                            "answer":  "Yes" if question == "Does robot arm position directly cause red light to change?" or 
                                    question == "Does robot arm position directly cause red light to change?" or 
                                    question == "Does robot arm position directly cause green light to change?" or
                                    question == "Does robot arm position directly cause blue light to change?" or 
                                    question == "Does blue light directly cause red light to change?" or
                                    question == "Does green light directly cause red light to change?" else "No"
                        })
                
        
        
        with open(f"../VL-ICL/new_data/{task}/{data}/query.json", "w") as f:
            # seen = set()
            # unique_data = []
            # comparison_keys = ['id', 'question'] 
            # for entry in jsondata_test:
            #     key = tuple(entry.get(k) for k in comparison_keys)  # Use tuple for hashable key

            #     if key not in seen:
            #         seen.add(key)
            #         unique_data.append(entry)
                    
            # jsondata_test = unique_data
            json.dump(jsondata_test, f, indent=4, ensure_ascii=False)
        
            # json.dump(jsondata_test, f, indent=4, ensure_ascii=False)
        
        with open(f"../VL-ICL/new_data/{task}/{data}/support.json", "w") as f:
            # seen = set()
            # unique_data = []
            # comparison_keys = ['id', 'question'] 
            
            # for entry in jsondata_train:
            #     key = tuple(entry.get(k) for k in comparison_keys)  # Use tuple for hashable key

            #     if key not in seen:
            #         seen.add(key)
            #         unique_data.append(entry)
                    
            # jsondata_train = unique_data
            json.dump(jsondata_train, f, indent=4, ensure_ascii=False)
        
main()
        
    