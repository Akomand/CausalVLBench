import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import json
import argparse
import gc
from utils import model_inference, utils, ICL_utils, load_models
from tqdm import tqdm
import random
import numpy as np
import time
import copy
# import deepspeed


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Inference')

    parser.add_argument('--dataDir', default='./', type=str, help='Data directory.')
    parser.add_argument('--dataset', default='pendulum', type=str, choices=["pendulum", "pendulum_small", "flow", "flow_small", "circuit"])
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-9b', 'idefics-80b-instruct', 'gemini', 'gemini_2_5', 'gpt4v', 'llava-onevision-7b', 'llava-onevision-72b', 'llava-onevision-7b-chat', 'llava-onevision-7b-ft',
                                                   'llava-onevision-0.5b', 'deepseek-vl2', 'deepseek-vl2-large', 'idefics2-8b', 'qwen-vl-2.5-instruct'],
                        default=["llava16-7b"], nargs="+")
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')

    parser.add_argument('--max_new_tokens', default=50, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='counterfactual', type=str, choices=['nothing', 'counterfactual', 'intervention', 'discovery', 'discovery_interleaved', 'intervention_nograph', 'counterfactual_nograph'], help='Detailed level of task description.')

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--run', default=1, type=int, help='Random seed.')
    parser.add_argument('--strategy', default="balanced", type=str, help='Selection strategy')
    parser.add_argument('--samples', default=1000, type=int, help='number of query samples')
    parser.add_argument('--sub_samples', default=0, type=int, help='number of query samples')
    parser.add_argument('--sub_flag', default=False, help='number of query samples')
    parser.add_argument('--cot', default=False, help='number of query samples')
        
    return parser.parse_args()


def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens

    
    client = None
        
    
    for i, query in enumerate(tqdm(query_meta)):
        
        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, query=query, selection_strategy=args.strategy)
        

        try:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                        n_shot_support, data_path, processor, max_new_tokens, client=client, use_cot=args.cot, idx=i)
            query['prediction'] = predicted_answer

            results.append(query)
        except Exception:
            continue

    return results
    

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.run)
    query_meta, support_meta = utils.load_data(args)
    
    if "discovery" in args.task_description:
        # num_ims = len(query_meta) / 12
        indices = np.arange(0, len(query_meta), 12).tolist()
        # indices = [i for i in range(query_meta)]
        sampled_indices = random.sample(indices, args.samples)
        # print(sampled_indices)
        # exit(0)
        query_meta_samples = []
        for i in range(len(sampled_indices)):
            query_meta_samples.append(query_meta[sampled_indices[i]:12+sampled_indices[i]])
        
        query_meta_samples = [ele for sublist in query_meta_samples for ele in sublist]
        
        query_meta = query_meta_samples
    else:
        query_meta = random.sample(query_meta, args.samples)
        if args.sub_flag:
            query_meta = query_meta[args.sub_samples*100:(args.sub_samples + 1)*100]
    
    # print(len(query_meta))
    # exit(0)
    
    for engine in args.engine:

        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        
        utils.set_random_seed(args.seed)
        for shot in args.n_shot:
            results_dict = eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, int(shot))
            os.makedirs(f"results/{args.task_description}/{args.dataset}", exist_ok=True)
            # if len(query_meta) < 2000:
            #     with open(f"results/{args.task_description}/{args.dataset}_small/{engine}_{shot}-shot.json", "w") as f:
            #         json.dump(results_dict, f, indent=4)
            # else:

            if args.sub_flag:
                with open(f"results/{args.task_description}/{args.dataset}/{engine}_{shot}-shot_{args.run}_{args.sub_samples}.json", "w") as f:
                    json.dump(results_dict, f, indent=4)
            elif args.cot:
                with open(f"results/{args.task_description}/{args.dataset}/{engine}_{shot}-shot_{args.run}_COT.json", "w") as f:
                    json.dump(results_dict, f, indent=4)
            elif args.strategy == "balanced":
                with open(f"results/{args.task_description}/{args.dataset}/{engine}_{shot}-shot_{args.run}_balanced.json", "w") as f:
                    json.dump(results_dict, f, indent=4)
            else:
                with open(f"results/{args.task_description}/{args.dataset}/{engine}_{shot}-shot_{args.run}.json", "w") as f:
                    json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()
