import numpy as np
import pandas as pd
import os
import json
from evals import eval
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams.update({'font.size': 40})

def main():
    seeds = [0, 1, 2]
    nshots = [0,2,4,8]
    # nshots = [0]
    models = [
        # 'llava-onevision-72b',
        # 'llava-onevision-7b',
        # 'qwen-vl-chat',
        # 'idefics2-8b', 
        # 'deepseek-vl2', 
        # 'openflamingo',
        # 'otter-llama',
        # 'deepseek-vl2-large', 
        # 'qwen-vl-2.5-instruct',
        'gemini',
        # 'gemini_2_5'
        ]

    model_names = {'gemini': 'Gemini-2.0-Flash',
                   'gemini_2_5': 'Gemini-2.5-Flash',
                   'llava-onevision-7b': "LLaVA-OneVision-7B",
                   'qwen-vl-2.5-instruct': "Qwen2.5-VL-32B-Instruct",
                   'qwen-vl-chat': "Qwen-VL-Chat",
                   'idefics2-8b': "IDEFICS2-8B",
                   'openflamingo': "OpenFlamingo",
                   'otter-llama': "Otter",
                   'deepseek-vl2': 'DeepseekVL2-Small',
                   'deepseek-vl2-large': 'DeepseekVL2'}
    
    # model_name_list = ['Gemini-2.0-Flash', "LLaVA-OneVision-7B", "Qwen2.5-VL-32B-Instruct", "Qwen-VL-Chat", "IDEFICS2-8B", "OpenFlamingo", "Otter", 'DeepseekVL2-Small', 'DeepseekVL2'] 
    
    datasets = [
        'pendulum', 
        'flow',
        'circuit'
        ]
    tasks = [
        # 'intervention',
        # 'intervention_nograph',
        'counterfactual',
        # 'counterfactual_nograph',
        # 'discovery_interleaved',
        # 'discovery'
        ]
    

    for task in tasks:
        print(f'\n---------------RESULTS FOR {task.upper()}---------------')
        if 'discovery' in task:
            nshots = [0]
        
        # count = 0
        for dataset in datasets:
            
            count = 0
            ######################   
            print(f'\n{dataset.upper()} DATASET')
            agg_res = {}
            agg_res_table = {}
            for model in models:
                # print(model_names[model])
                # exit(0)
                agg_res[model_names[model]] = []
                agg_res_table[model] = []
                path = f'results/{task}/{dataset}/{model}'
                for nshot in nshots:
                    avg_std_dev = agg_res_seeds(path, dataset, nshot, seeds, model)
                    # print(avg_std_dev)
                    agg_res[model_names[model]].append(avg_std_dev[0][0])
                    
                    agg_res_table[model].extend(avg_std_dev)
                    # print()
            #######################
            
            count += 1
            agg_df = pd.DataFrame.from_dict(agg_res_table, orient='index')
            if 'discovery' in task:
                agg_df.columns = ['shd', 'acc']
            else:
                agg_df.columns = [str(a) for a in nshots]
            agg_df_pretty_print = agg_df.map(lambda x: f'{"{:.2f}".format(x[0])}+-{"{:.2f}".format(x[1])}')
            agg_df_pretty_print = agg_df_pretty_print.reset_index(names='Model')
            print(agg_df_pretty_print)
            print()
            agg_df = agg_df.map(lambda x: f'${"{:.1f}".format(x[0])}_{{{"{:.1f}".format(x[1])}}}$')
            # agg_df = agg_df.map(lambda x: f'${"{:.2f}".format(x[0])}$')
            agg_df = agg_df.reset_index(names='Model')
            print(agg_df.to_latex(index=False,
                                bold_rows=True,
            ))  

            
def agg_res_seeds(path, dataset, nshot, seeds=[0,1,2,3,4], model_n = None):
    agg_res = []
    agg_shd = []
    
    count_n = {'pendulum angle': 0,
                'light position': 0,
                'shadow length': 0,
                'shadow position': 0,}
    # count_n = {'ball size': 0,
    #             'hole position': 0,
    #             'water level': 0}
    # count_n = {'robot arm': 0,
    #             'green light': 0,
    #             'red light': 0,
    #             'blue light': 0}
    
    for seed in seeds:
        result_file = f"{path}_{nshot}-shot_{seed}_COT.json"
        try:
            with open(result_file, "r") as f:
                results_dict = json.load(f)
            
            if 'intervention' in path:
                scores = eval.exact_match_interv(results_dict, dataset, nshot)
                # for k, v in count.items():
                #     count_n[k] += count[k] 
            elif 'counterfactual' in path:
                scores = eval.exact_match_cf(results_dict, dataset, nshot, model_n=model_n)
            elif 'discovery' in path:
                shd, scores = eval.exact_match_discovery(results_dict, dataset, nshot)

                
                # exit(0)
                # pass
                agg_shd.append(shd)
            agg_res.append(scores*100.0)
        except:
            print(f'{result_file} does not exist!')

    avg = np.mean(agg_res)
    std_dev = np.std(agg_res)
    
    # if 'intervention' in path:
    #     for k, v in count_n.items():
    #         count_n[k] = count_n[k] / 3
        
    #     return count_n
            
    
    if 'discovery' in path:
        avg_shd = np.mean(agg_shd)
        shd_std_dev = np.std(agg_shd)
        
        return [[avg_shd, shd_std_dev], [avg, std_dev]]
    
    return [[avg, std_dev]]
        
               
    
if __name__=="__main__":
    main()