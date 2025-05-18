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
        'llava-onevision-7b',
        'qwen-vl-chat',
        'idefics2-8b', 
        'deepseek-vl2', 
        'openflamingo',
        'otter-llama',
        'deepseek-vl2-large', 
        'qwen-vl-2.5-instruct',
        'gemini'
        ]

    model_names = {'gemini': 'Gemini-2.0-Flash',
                   'llava-onevision-7b': "LLaVA-OneVision-7B",
                   'qwen-vl-2.5-instruct': "Qwen2.5-VL-32B-Instruct",
                   'qwen-vl-chat': "Qwen-VL-Chat",
                   'idefics2-8b': "IDEFICS2-8B",
                   'openflamingo': "OpenFlamingo",
                   'otter-llama': "Otter",
                   'deepseek-vl2': 'DeepseekVL2-Small',
                   'deepseek-vl2-large': 'DeepseekVL2'}
    
    full_dataset = {
        'Pendulum': 'Pendulum',
        'Flow': 'Water Flow',
        'Circuit': 'Causal Circuit'
    }
    
    # model_name_list = ['Gemini-2.0-Flash', "LLaVA-OneVision-7B", "Qwen2.5-VL-32B-Instruct", "Qwen-VL-Chat", "IDEFICS2-8B", "OpenFlamingo", "Otter", 'DeepseekVL2-Small', 'DeepseekVL2'] 
    
    datasets = [
        # 'pendulum', 
        # 'flow',
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
    
    pend_variables = {  
        'pendulum angle': ['shadow length', 'shadow position'],
        'light position': ['shadow length', 'shadow position'],
    }
    
    flow_variables = {
        'ball size': ['water level', 'water flow'],
        'hole position': ['water flow'],
        'water level': ['water flow'],
    }
    
    circuit_variables = {
        'robot arm': ['green light, blue light', 'red light'],
        'green light': ['red light'],
        'blue light': ['red light'],
    }
    
    visualize = True
    for task in tasks:
        print(f'\n---------------RESULTS FOR {task.upper()}---------------')
        if 'discovery' in task:
            nshots = [0]
        
        
        if task == "counterfactual":
            fig, axes = plt.subplots(1, 3, figsize=(16,4))  # 1 rows, 3 columns
            # fig, axes = plt.subplots(2, 4, figsize=(32,8))  # 1 rows, 3 columns
        
        # count = 0
        for dataset in datasets:
            
            if dataset == 'pendulum':
                variables = pend_variables
            elif dataset == 'flow':
                variables = flow_variables
            else:
                variables = circuit_variables
            
            fig.suptitle(f'Counterfactual Prediction on {full_dataset[dataset.capitalize()]} Dataset\n', fontsize=18)
            count = 0
            for key, item in variables.items():
                print(key, item)
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
                        avg_std_dev = agg_res_seeds(path, dataset, nshot, seeds, model, var=[key, item])
                        # print(avg_std_dev)
                        agg_res[model_names[model]].append(avg_std_dev[0][0])
                        
                        agg_res_table[model].extend(avg_std_dev)
                        # print()
                #######################
                if visualize:
                    # Convert to long-form DataFrame for seaborn
                    data = []
                    for model, accs in agg_res.items():
                        for shot, acc in zip(nshots, accs):
                            data.append({"Number of shots": str(shot), "Accuracy (%)": acc, "Model": model})

                    df = pd.DataFrame(data)

                    if key == 'pendulum angle':
                        count1 = 0
                        count2 = 0
                    elif key == 'light position':
                        count1 = 0
                        count2 = 1
                    elif key == 'ball size':
                        count1 = 0
                        count2 = 2
                    elif key == 'hole position':
                        count1 = 0
                        count2 = 3
                    elif key == 'water level':
                        count1 = 1
                        count2 = 0
                    elif key == 'robot arm':
                        count1 = 1
                        count2 = 1
                    elif key == 'green light':
                        count1 = 1
                        count2 = 2
                    elif key == 'blue light':
                        count1 = 1
                        count2 = 3
                        
                    # Plot using seaborn
                    sns.set(style="white")
                    
                    # plt.figure(figsize=(4, 8))
                    if task == "counterfactual":
                        sns.lineplot(data=df, x="Number of shots", y="Accuracy (%)", hue="Model", marker="o", markersize=10, ax=axes[count], linewidth=3, palette="bright", alpha=0.75)
                        # sns.lineplot(data=df, x="Number of shots", y="Accuracy (%)", hue="Model", marker="o", markersize=8, ax=axes[count1][count2])
                    # elif task == "counterfactual":
                    #     plt.figure(figsize=(20, 4))
                    #     sns.barplot(data=df, x="Model", y="Accuracy (%)", hue="Number of shots", dodge=True, width=0.5)
                    
                    sns.despine(top=True, right=True)
                    
                    # if task == "counterfactual":
                    #     axes[count1][count2].get_legend().remove()
                    #     axes[count1][count2].set_xlabel("Number of shots", fontsize=16)
                    #     axes[count1][count2].set_ylabel("Accuracy (%)", fontsize=16)
                    #     # axes[count].set_title(f'{datasets[count]} Intervention Prediction', fontsize=16)
                    #     axes[count1][count2].set_title(f'{dataset.capitalize()} Counterfactual Prediction\n({key} intervention)', fontsize=16)

                    #     axes[count1][count2].tick_params(axis='both', labelsize=16)
                    
                    if task == "counterfactual":
                        axes[count].get_legend().remove()
                        axes[count].set_xlabel("Number of shots", fontsize=16)
                        axes[count].set_ylabel("Accuracy (%)", fontsize=16)
                        # axes[count].set_title(f'{datasets[count]} Intervention Prediction', fontsize=16)
                        axes[count].set_title(f'{key.title()} Intervention', fontsize=16)

                        axes[count].tick_params(axis='both', labelsize=16)
                        
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
                # agg_df = agg_df.map(lambda x: f'${"{:.1f}".format(x[0])}_{{{"{:.1f}".format(x[1])}}}$')
                agg_df = agg_df.map(lambda x: f'${"{:.2f}".format(x[0])}$')
                agg_df = agg_df.reset_index(names='Model')
                print(agg_df.to_latex(index=False,
                                    bold_rows=True,
                ))  

        if task == "counterfactual":
            plt.xticks(['0', '2', '4', '8'])
        
        if task == "counterfactual": 
            handles, labels = axes[0].get_legend_handles_labels()
            # handles, labels = axes[0][0].get_legend_handles_labels()
            plt.legend(handles, labels, ncol=4, bbox_to_anchor=(1.0, -0.3), fontsize=16)#, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=5)

        # if task == "counterfactual":
        #     plt.legend(loc='upper right', fontsize=12)
        
        if visualize:
            # plt.savefig(f"accuracy_vs_shot_cf_pendulum_angle.pdf", format="pdf", bbox_inches='tight', dpi=300)
            plt.subplots_adjust(hspace=1)
            fig.subplots_adjust(top=0.8)
            # plt.savefig(f"accuracy_vs_shot_cf_flow.png", bbox_inches='tight', dpi=300)

            plt.savefig(f"accuracy_vs_shot_cf_circuit.pdf", format="pdf", bbox_inches='tight', dpi=300)
            plt.show()
            
def agg_res_seeds(path, dataset, nshot, seeds=[0,1,2,3,4], model_n = None, var=None):
    agg_res = []
    agg_shd = []
    for seed in seeds:
        result_file = f"{path}_{nshot}-shot_{seed}.json"
        try:
            with open(result_file, "r") as f:
                results_dict = json.load(f)
 
            if 'intervention' in path:
                scores = eval.exact_match_interv(results_dict, dataset, nshot)
            elif 'counterfactual' in path:
                scores = eval.exact_match_cf_decomp(results_dict, dataset, nshot, model_n=model_n, var=var)
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
    
    if 'discovery' in path:
        avg_shd = np.mean(agg_shd)
        shd_std_dev = np.std(agg_shd)
        
        return [[avg_shd, shd_std_dev], [avg, std_dev]]
    
    return [[avg, std_dev]]
        
               
    
if __name__=="__main__":
    main()