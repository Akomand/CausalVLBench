import numpy as np
import re
import torch
import os
import json
from collections import defaultdict
from cdt.metrics import SHD
from copy import deepcopy

def eval_scores(results, dataset, model=None, tokenizer=None, processor=None, n_shot=4):
    if dataset in ['pendulum', 'pendulum_small', 'flow', 'flow_small', 'circuit', 'circuit_small']:
        score = exact_match_discovery(results, dataset, n_shot)
        
    return score

def exact_yes_no(results):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if result['answer'].lower() == 'yes' and 'yes' in str(prediction).lower():
            acc.append(1)
        elif result['answer'].lower() == 'no' and 'yes' not in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_in_match(results):
    acc = []
    for result in results:
        prediction = result['prediction'][0].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        # if str(result['answer']).lower() in str(prediction).lower():
        if str(prediction).lower() in str(result['answer']).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc


# Evaluation for counterfactual task
def exact_match_discovery(results, dataset, nshot=4):    
    # if isinstance(result['prediction'], str):
    #         prediction = result['prediction'].strip()
    # else:
    #     prediction = result['prediction'][0].strip()
        
    accuracy = []
    # print(len(results))
    # exit(0)
    
    grouped_objects = defaultdict(list)
    count = defaultdict(int)
    for result in results:
        if isinstance(result['prediction'], str):
            result['prediction'] = result['prediction'].strip()
        else:
            result['prediction'] = result['prediction'][0].strip()
            
        grouped_objects[result['id']].append(result)
    # print(len(grouped_objects))
    # exit(0)
    for result in results:
        if len(grouped_objects[result['id']]) < 12:
            del grouped_objects[result['id']]
            # grouped_objects[result['id']] = grouped_objects[result['id']][:12]
        
    if dataset == "pendulum":
        ground_truth_causal_graph = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(4, 4)
    elif dataset == "flow":
        ground_truth_causal_graph = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(4, 4)
    elif dataset == "circuit":
        ground_truth_causal_graph = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(4, 4)
    
    # ground_truth_answers = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    hamming = defaultdict(list)
    acc = 0.0
    acc_arr = []
    for key, value in grouped_objects.items():
        ground_truth_answers = [1 if val['answer'].lower().strip() == "yes" else 0 for val in value]
        adj_list = [1 if val['prediction'].lower().strip() == "yes" else 0 for val in value]
        temp = np.array(deepcopy(adj_list))
        

        adj_list.insert(0, 0)
        adj_list.insert(5, 0)
        adj_list.insert(10, 0)
        adj_list.insert(15, 0)
        
        
        adj_matrix = np.array(adj_list).reshape(4, 4)
        
        # Compute SHD
        shd = SHD(ground_truth_causal_graph, adj_matrix, double_for_anticausal=False)

        hamming[key] = shd
        # print(ground_truth_answers)
        # print(temp)
        # print((ground_truth_answers == temp))
        # print(key)
        # print(value)
        # exit(0)
        acc += (ground_truth_answers == temp).mean()
        acc_arr.append(temp.tolist())
    
    avg_shd_ls = []
    # print(hamming)
    for k, v in hamming.items():
        avg_shd_ls.append(v)

        
    avg_acc = acc / len(hamming.items())

    # for i in range(5):
    #     print(acc_arr[i])
    # print()
    # print(f'Avg. SHD: {sum(avg_shd_ls) / len(avg_shd_ls)}')
    # print(f'Avg. Acc: {avg_acc}')
    # exit(0)
    avg_shd = sum(avg_shd_ls) / len(avg_shd_ls)
    
    return avg_shd, avg_acc


maps = {
    'shadow length': 'shadow_len_cf',
    'shadow position': 'shadow_pos_cf',
    'water level': 'water_level_cf',
    'water flow': 'water_flow_cf',
    'green light': 'green light cf',
    'blue light': 'blue light cf',
    'red light': 'red light cf',
}

# Evaluation for counterfactual task
def exact_match_cf_decomp(results, dataset, nshot=4, model_n=None, var=None):    
    accuracy = []
    
    k, items = var
    for result in results:

        if result['intervention'] == k:
            if isinstance(result['prediction'], str):
                prediction = result['prediction'].strip()
            else:
                prediction = result['prediction'][0].strip()
            ans_dict = {}
            
            prediction = prediction.replace("*", "")
            answers_list = prediction.lower().split('\n')
            
            if model_n == 'qwen-vl-chat':
                sample_acc = 0.0
                if 'pendulum' in dataset:
                    for i in items:
                        if f"{i}: {result[maps[i]]}" in result["prediction"]:
                            sample_acc += 1.0
                elif 'flow' in dataset:
                    for i in items:
                        if f"{i}: {result[maps[i]]}" in result["prediction"]:
                            sample_acc += 1.0
                elif 'circuit' in dataset:
                    for i in items:
                        if f"{i}: {result[maps[i]]}" in result["prediction"]:
                            sample_acc += 1.0
                            
            for answer in answers_list:
                ans = answer.split(":")
                if len(ans) > 1:
                    ans_dict[ans[0].strip()] = ans[1].strip()
            
            if model_n != 'qwen-vl-chat':
                sample_acc = 0.0
                for key, val in ans_dict.items():
                    
                                
                    # print(ans_dict[key])
                    # exit(0)
                    if 'circuit' in dataset:
                        for i in items:
                            if i in key:
                                if result[maps[i]] == ans_dict[key]:
                                    sample_acc += 1.0
                    elif 'pendulum' in dataset:
                        for i in items:
                            if i in key:
                                if result[maps[i]] == ans_dict[key]:
                                    sample_acc += 1.0
                    elif 'flow' in dataset:
                        for i in items:
                            if i in key:
                                if result[maps[i]] == ans_dict[key]:
                                    sample_acc += 1.0
                            
            sample_acc = sample_acc / len(items)
            accuracy.append(sample_acc)
                
    avg_acc = sum(accuracy) / len(accuracy)
    
    return avg_acc



# Evaluation for counterfactual task
def exact_match_cf(results, dataset, nshot=4, model_n=None, var=None):    
    accuracy = []
    for result in results:

        
        if isinstance(result['prediction'], str):
            prediction = result['prediction'].strip()
        else:
            prediction = result['prediction'][0].strip()
        ans_dict = {}
        
        prediction = prediction.replace("*", "")
        answers_list = prediction.lower().split('\n')
        
        if model_n == 'qwen-vl-chat':
            sample_acc = 0.0
            if 'pendulum' in dataset:
                if f"pendulum angle: {result['angle_cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"light position: {result['light_cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"shadow length: {result['shadow_len_cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"shadow position: {result['shadow_pos_cf']}" in result["prediction"]:
                    sample_acc += 1.0
            elif 'flow' in dataset:
                if f"ball size: {result['ball_cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"hole position: {result['hole_cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"water level: {result['water_level_cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"water flow: {result['water_flow_cf']}" in result["prediction"]:
                    sample_acc += 1.0
            elif 'circuit' in dataset:
                if f"red light: {result['red light cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"green light: {result['green light cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"blue light: {result['blue light cf']}" in result["prediction"]:
                    sample_acc += 1.0
                if f"robot arm: {result['robot arm cf']}" in result["prediction"]:
                    sample_acc += 1.0
                        
        for answer in answers_list:
            ans = answer.split(":")
            if len(ans) > 1:
                ans_dict[ans[0].strip()] = ans[1].strip()
        
        if model_n != 'qwen-vl-chat':
            sample_acc = 0.0
            for key, val in ans_dict.items():
                if 'circuit' in dataset:
                    if 'red' in key:
                        if (result['red light cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'green' in key:
                        if(result['green light cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'blue' in key:
                        if(result['blue light cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'robot' in key or 'arm' in key:
                        if(result['robot arm cf'] == ans_dict[key]):
                            sample_acc += 1.0
                elif 'pendulum' in dataset:
                    if 'pendulum angle' in key:
                        if (result['angle_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'light position' in key:
                        if(result['light_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    if 'shadow length' in key:
                        if(result['shadow_len_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'shadow position' in key:
                        if(result['shadow_pos_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                elif 'flow' in dataset:
                    if 'ball' in key:
                        if (result['ball_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'hole' in key:
                        if(result['hole_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    if 'level' in key:
                        if(result['water_level_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                    elif 'flow' in key:
                        if(result['water_flow_cf'] == ans_dict[key]):
                            sample_acc += 1.0
                        
        sample_acc = sample_acc / 4.0
        accuracy.append(sample_acc)
                
    avg_acc = sum(accuracy) / len(accuracy)
    
    return avg_acc

# evaluation for 
def exact_match_interv(results, dataset, nshot=4):
    acc = []
    
    if "pendulum" in dataset:
        count = {'pendulum angle': 0,
                'light position': 0,
                'shadow length': 0,
                'shadow position': 0,
                'random': 0}
        
        correct_count = {'pendulum angle': 0,
                'light position': 0,
                'shadow length': 0,
                'shadow position': 0}
        
    elif "flow" in dataset:
        count = {'ball size': 0,
                'hole position': 0,
                'water level': 0,
                'random': 0}

        correct_count = {'ball size': 0,
                'hole position': 0,
                'water level': 0}
        
    elif "circuit" in dataset:
        count = {'robot arm': 0,
                'green light': 0,
                'red light': 0,
                'blue light': 0}

        correct_count = {'robot arm': 0,
                        'green light': 0,
                        'red light': 0,
                        'blue light': 0}

    for result in results:
        if isinstance(result['prediction'], str):
            prediction = result['prediction'].strip()
        else:
            prediction = result['prediction'][0].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if 'operator_induction' in dataset or 'clevr_simple' in dataset:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                prediction = ''

        if nshot > 0:

            if str(result['answer']).lower() in str(prediction.split('.')[0]).lower():
                acc.append(1)
                correct_count[str(result['answer']).lower()] += 1
            else:
                acc.append(0)
                
            for k,v in count.items():
                if k in str(prediction).lower():
                    count[k] += 1    
        elif nshot == 0:
            if dataset == 'circuit' and 'arm position' in str(prediction).lower():
                prediction = 'robot arm'
                
            if str(result['answer']).lower() in str(prediction).lower():
                acc.append(1)
                correct_count[str(result['answer']).lower()] += 1
            else:
                acc.append(0)
                
            for k,v in count.items():
                if k in str(prediction).lower():
                    count[k] += 1
        
    # for k,v in  count.items():
    #     print(f'{k}: {v}')
    
    # for k,v in  correct_count.items():
    #     print(f'{k}: {v}')
    
    avg_acc = np.average(acc)
    return avg_acc
