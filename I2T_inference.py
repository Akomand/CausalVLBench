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
    parser.add_argument('--dataset', default='operator_induction', type=str, choices=['operator_induction', 'textocr', 'open_mi', 
                                                                             'clevr','operator_induction_interleaved', 'matching_mi', "pendulum", "pendulum_small", "flow", "flow_small", "circuit"])
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

def e_prepare_deepspeed(model, accelerator):
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
    
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    config_kwargs["optimizer"] = {"type": None}

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    #set the gradients to false for every parameter
    for param in model.parameters():
        param.requires_grad = False
        
    print(f'Loaded model with deepspeed ZeRO stage-{config_kwargs["zero_optimization"]["stage"]}')
    
    return model

def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens

    if 'gemini' in engine:
        time.sleep(61)
        from google import genai
            
        api_keys = {
            'counterfactual': {
                'pendulum_0': ['AIzaSyDcHUe4GlGgKSBMnKUrIN1lteODSyNMd60'],
                'flow_0': ['AIzaSyAZB3rV9FzAYwOJyBdlOgFHmAmZxmnPk44'],
                'circuit_0': ['AIzaSyDUJxhVmspRUDqWKm6-vA7Afxzd4ArjEd0'],
            },
            'counterfactual_replacement': {
                'pendulum_0': ['AIzaSyAaVD8ACd-wp747oSr_qGxUn7WkKw02XUo'],
                'flow_0': ['AIzaSyB9mSXTunsnrBJeaeDQN3x1csZcVvX1sFw'],
                'circuit_0': ['AIzaSyDk_z72r3qpATSthV2v8exQRPMMWwYX9iE'],
            },
            # 'intervention': {
            #     'pendulum_0': ['AIzaSyBEiEP9PI4y6AY0luXYXZWAf9VIZ0RL74A'],
            #     'flow_0': ['AIzaSyBuDVxIOqlFycG-_3HNtAJUfBJe66pOlo0'],
            #     'circuit_0': ['AIzaSyC_q_ESiUAyWl8lodQfqt5X1Gt2G2GvXI0'],
            # },
            # 'intervention_replacement': {
            #     'pendulum_0': ['AIzaSyDUZebgvr-OpoWzGYxi7nxwoALiZ4ewWlk'],
            #     'flow_0': ['AIzaSyC6IOpvlHNBWNYRdvNZyJnbHTqxsAHOMO0'],
            #     'circuit_0': ['AIzaSyD5cxafkZZlYKTdcWWfHIX50dm9sXu4ric'],
            # }   
        }
        # api_keys = {
        #     #akomanduri124
        #     'discovery_interleaved': {
        #         'pendulum_0': ['AIzaSyCjqDPuonjK0XwDOtG_RxDrQsaIKUBISgE'],
        #         'pendulum_2': ['AIzaSyBxcFmFTWerHcCCqOGtUiCyzRzyxp1aBF0', 'AIzaSyByax8Cwd0yi1XMLDCerzNi0GMxZXG9Sew'],
        #         'pendulum_4': ['AIzaSyByax8Cwd0yi1XMLDCerzNi0GMxZXG9Sew'],
        #         'pendulum_8': ['AIzaSyCjqDPuonjK0XwDOtG_RxDrQsaIKUBISgE'],
        #         'flow_0': ['AIzaSyBRDq_qGluiOAYa2je0IFQiHKxarqLnFtQ'],
        #         'flow_2': ['AIzaSyAdiQjcSi70kPyoZ_qSfNjknE2nHvFpXuI', 'AIzaSyB-aYBTwPHKfV3tZDUgMxU12d9_-7A9W04'],
        #         'flow_4': ['AIzaSyB-aYBTwPHKfV3tZDUgMxU12d9_-7A9W04'],
        #         'flow_8': ['AIzaSyBRDq_qGluiOAYa2je0IFQiHKxarqLnFtQ'],
        #         'circuit_0': ['AIzaSyBNyxnprFWA1_zPtbUKobfABqOEkiHMTVI'],
        #         'circuit_2': ['AIzaSyA1-PKUHDyL-g9fnUORTcLR3um1UXISePw', 'AIzaSyA1-PKUHDyL-g9fnUORTcLR3um1UXISePw'],
        #         'circuit_4': ['AIzaSyBOOmRV6o3zjKCpuIDR5pSx7qjc_3Gtqj8'],
        #         'circuit_8': ['AIzaSyBNyxnprFWA1_zPtbUKobfABqOEkiHMTVI'],
        #         'circuit_4': ["AIzaSyBuDVxIOqlFycG-_3HNtAJUfBJe66pOlo0"],
        #         'circuit_8': ["AIzaSyBEiEP9PI4y6AY0luXYXZWAf9VIZ0RL74A"]
        #     },
            
        #     #karunabhaila100
        #     'discovery_interleaved_replacement': {
        #         'pendulum_0': ["AIzaSyAzgSX_lRWOnGwj7F1p4VqYh_9BpcxPdLI"],
        #         'pendulum_2': ["AIzaSyANAZvBatavNuE1no3qpbaTsOlPct1v9nU"],
        #         'pendulum_4': ["AIzaSyANAZvBatavNuE1no3qpbaTsOlPct1v9nU"],
        #         'pendulum_8': ["AIzaSyAzgSX_lRWOnGwj7F1p4VqYh_9BpcxPdLI"],
        #         'flow_0': ["AIzaSyC6IOpvlHNBWNYRdvNZyJnbHTqxsAHOMO0"],
        #         'flow_2': ['AIzaSyA1-PKUHDyL-g9fnUORTcLR3um1UXISePw'],
        #         'flow_4': ["AIzaSyD5cxafkZZlYKTdcWWfHIX50dm9sXu4ric"],
        #         'flow_8': ["AIzaSyC6IOpvlHNBWNYRdvNZyJnbHTqxsAHOMO0"],
        #         'circuit_0': ["AIzaSyBEiEP9PI4y6AY0luXYXZWAf9VIZ0RL74A"],
        #         'circuit_2': ['AIzaSyB-aYBTwPHKfV3tZDUgMxU12d9_-7A9W04'],
        #         'circuit_4': ["AIzaSyBuDVxIOqlFycG-_3HNtAJUfBJe66pOlo0"],
        #         'circuit_8': ["AIzaSyBEiEP9PI4y6AY0luXYXZWAf9VIZ0RL74A"]       
        #     }
        # }
        
        
        # api_keys = {
        #     #akomanduri124
        #     'counterfactual': {
        #         'pendulum_0': ['AIzaSyDUJxhVmspRUDqWKm6-vA7Afxzd4ArjEd0'],
        #         # 'pendulum_2': ['AIzaSyBBR4jZ5YTwQAnWwiaAWbJGbAcDEOCXdO0'],
        #         'pendulum_4': ['AIzaSyAZB3rV9FzAYwOJyBdlOgFHmAmZxmnPk44'],
        #         'pendulum_8': ['AIzaSyDUJxhVmspRUDqWKm6-vA7Afxzd4ArjEd0'],
        #         'flow_0': ['AIzaSyB9mSXTunsnrBJeaeDQN3x1csZcVvX1sFw'],
        #         # 'flow_2': ['AIzaSyCD_Tn-JJv6C4Bcz-uXafKTquncnwecgGg'],
        #         'flow_4': ['AIzaSyCD_Tn-JJv6C4Bcz-uXafKTquncnwecgGg'],
        #         'flow_8': ['AIzaSyB9mSXTunsnrBJeaeDQN3x1csZcVvX1sFw'],
        #         'circuit_0': ['AIzaSyCkqd3ekWmcdUDvH3H_Ir7h20eFZZXHPGk'],
        #         # 'circuit_2': ['AIzaSyDk_z72r3qpATSthV2v8exQRPMMWwYX9iE'],
        #         'circuit_4': ['AIzaSyAiVIsjBMtGAL_M4gK0Gyxa7tvIkJRPZxA'],
        #         'circuit_8': ['AIzaSyCkqd3ekWmcdUDvH3H_Ir7h20eFZZXHPGk']
        #     },
        #     #karunabhaila100
        #     'counterfactual_replacement': {
        #         'pendulum_0': ['AIzaSyBNyxnprFWA1_zPtbUKobfABqOEkiHMTVI'],
        #         'pendulum_2': ["AIzaSyANAZvBatavNuE1no3qpbaTsOlPct1v9nU"],
        #         'pendulum_4': ['AIzaSyBxcFmFTWerHcCCqOGtUiCyzRzyxp1aBF0'],
        #         'pendulum_8': ['AIzaSyBNyxnprFWA1_zPtbUKobfABqOEkiHMTVI'],
        #         'flow_0': ['AIzaSyB-aYBTwPHKfV3tZDUgMxU12d9_-7A9W04'],
        #         'flow_2': ["AIzaSyD5cxafkZZlYKTdcWWfHIX50dm9sXu4ric"],
        #         'flow_4': ['AIzaSyAdiQjcSi70kPyoZ_qSfNjknE2nHvFpXuI'],
        #         'flow_8': ['AIzaSyB-aYBTwPHKfV3tZDUgMxU12d9_-7A9W04'],
        #         'circuit_0': ['AIzaSyA1-PKUHDyL-g9fnUORTcLR3um1UXISePw'],
        #         'circuit_2': ["AIzaSyC_q_ESiUAyWl8lodQfqt5X1Gt2G2GvXI0"],
        #         'circuit_4': ['AIzaSyBqZ3ZyLYWESepjy_DaoG_t9IVANC7iqV8'],
        #         'circuit_8': ['AIzaSyA1-PKUHDyL-g9fnUORTcLR3um1UXISePw']       
        #     },
            
        #     'discovery': {
                
        #     }
        # }
        
        # api_keys = {
        #     #kvmoon19
        #     'intervention': {
        #         'pendulum_0': ['AIzaSyDcHUe4GlGgKSBMnKUrIN1lteODSyNMd60'],
        #         # 'pendulum_2': ['AIzaSyBBR4jZ5YTwQAnWwiaAWbJGbAcDEOCXdO0'],
        #         'pendulum_4': ['AIzaSyAZB3rV9FzAYwOJyBdlOgFHmAmZxmnPk44'],
        #         'pendulum_8': ['AIzaSyDUJxhVmspRUDqWKm6-vA7Afxzd4ArjEd0'],
        #         'flow_0': ['AIzaSyAaVD8ACd-wp747oSr_qGxUn7WkKw02XUo'],
        #         # 'flow_2': ['AIzaSyCD_Tn-JJv6C4Bcz-uXafKTquncnwecgGg'],
        #         'flow_4': ['AIzaSyCD_Tn-JJv6C4Bcz-uXafKTquncnwecgGg'],
        #         'flow_8': ['AIzaSyB9mSXTunsnrBJeaeDQN3x1csZcVvX1sFw'],
        #         'circuit_0': ['AIzaSyBBR4jZ5YTwQAnWwiaAWbJGbAcDEOCXdO0'],
        #         # 'circuit_2': ['AIzaSyDk_z72r3qpATSthV2v8exQRPMMWwYX9iE'],
        #         'circuit_4': ['AIzaSyAiVIsjBMtGAL_M4gK0Gyxa7tvIkJRPZxA'],
        #         'circuit_8': ['AIzaSyCkqd3ekWmcdUDvH3H_Ir7h20eFZZXHPGk']
        #     },
            
        #     #alternateacc
        #     'intervention_replacement': {
        #         'pendulum_0': ['AIzaSyD5CILfjYVfkrLIt9BK3eoXomLBOe-bO_k'],
        #         'pendulum_2': ['AIzaSyAlVpk9IWOxgLE9RYOmAR4KPg85gldEaBs'],
        #         'pendulum_4': ['AIzaSyDWQ01DT8_x-Jxpw3tvrvRtYJM9F-JyJUg'],
        #         'pendulum_8': ['AIzaSyAiVIsjBMtGAL_M4gK0Gyxa7tvIkJRPZxA'],#,'AIzaSyAEeoSJI9Nytb3alK5_PKUbo041RC-GjSU'],
        #         'flow_0': ['AIzaSyDa2c3Xt3IqllAOOScIi5_1xAViWNMVcus'],
        #         'flow_2': ['AIzaSyDUJxhVmspRUDqWKm6-vA7Afxzd4ArjEd0'],#,['AIzaSyAsz0czIheKaQGCIUc2IHwyVFklXHNeyw0'],
        #         # 'flow_4': ["AIzaSyD5cxafkZZlYKTdcWWfHIX50dm9sXu4ric"],
        #         'flow_8': ['AIzaSyBRDq_qGluiOAYa2je0IFQiHKxarqLnFtQ'],
        #         'circuit_0': ['AIzaSyAUdPhKJZoV6YbJb53j3D3HhBE2NszT-y0'],
        #         'circuit_2': ['AIzaSyBPdm_5aol47Wie0ppxj8SJEUGQe5Wur5I'], 
        #         'circuit_4': ['AIzaSyC7BkkZtsk_pEJOj1nKFGf0pr7uVehLOr4'],
        #         'circuit_8': ['AIzaSyC_q_ESiUAyWl8lodQfqt5X1Gt2G2GvXI0'],
        #     },
            
        #     'discovery': {
                
        #     }
        # }
        
        
        # akomanduri124
        # api_keys1 = ['AIzaSyBxcFmFTWerHcCCqOGtUiCyzRzyxp1aBF0',
        # 'AIzaSyByax8Cwd0yi1XMLDCerzNi0GMxZXG9Sew',
        # 'AIzaSyCjqDPuonjK0XwDOtG_RxDrQsaIKUBISgE',
        # 'AIzaSyAdiQjcSi70kPyoZ_qSfNjknE2nHvFpXuI',
        # 'AIzaSyB-aYBTwPHKfV3tZDUgMxU12d9_-7A9W04',
        # 'AIzaSyBRDq_qGluiOAYa2je0IFQiHKxarqLnFtQ',
        # 'AIzaSyBqZ3ZyLYWESepjy_DaoG_t9IVANC7iqV8',
        # 'AIzaSyA1-PKUHDyL-g9fnUORTcLR3um1UXISePw',
        # 'AIzaSyBOOmRV6o3zjKCpuIDR5pSx7qjc_3Gtqj8',
        # 'AIzaSyBNyxnprFWA1_zPtbUKobfABqOEkiHMTVI']
        
        # karunabhaila100
        # api_keys2 = ['AIzaSyBEiEP9PI4y6AY0luXYXZWAf9VIZ0RL74A',
        # 'AIzaSyBuDVxIOqlFycG-_3HNtAJUfBJe66pOlo0',
        # 'AIzaSyC_q_ESiUAyWl8lodQfqt5X1Gt2G2GvXI0',
        # 'AIzaSyDUZebgvr-OpoWzGYxi7nxwoALiZ4ewWlk',
        # 'AIzaSyC6IOpvlHNBWNYRdvNZyJnbHTqxsAHOMO0',
        # 'AIzaSyD5cxafkZZlYKTdcWWfHIX50dm9sXu4ric',
        # 'AIzaSyBXi8fOMqaT_-unAw4aEQzzwqk5FI5AY9M',
        # 'AIzaSyAzgSX_lRWOnGwj7F1p4VqYh_9BpcxPdLI',
        # 'AIzaSyANAZvBatavNuE1no3qpbaTsOlPct1v9nU',
        # 'AIzaSyAVEFPhgPe-ccV4iH5GqUNtV_DplgfwBd0']
        
        # #kvmoon19
        # api_keys3 = ['AIzaSyDcHUe4GlGgKSBMnKUrIN1lteODSyNMd60',
        #         'AIzaSyAZB3rV9FzAYwOJyBdlOgFHmAmZxmnPk44',
        #         'AIzaSyDUJxhVmspRUDqWKm6-vA7Afxzd4ArjEd0',
        #         'AIzaSyAaVD8ACd-wp747oSr_qGxUn7WkKw02XUo',
        #         'AIzaSyB9mSXTunsnrBJeaeDQN3x1csZcVvX1sFw',
        #         'AIzaSyDk_z72r3qpATSthV2v8exQRPMMWwYX9iE',
        #         'AIzaSyAiVIsjBMtGAL_M4gK0Gyxa7tvIkJRPZxA',
        #         'AIzaSyCkqd3ekWmcdUDvH3H_Ir7h20eFZZXHPGk',
        #         'AIzaSyCD_Tn-JJv6C4Bcz-uXafKTquncnwecgGg',
        #         'AIzaSyBBR4jZ5YTwQAnWwiaAWbJGbAcDEOCXdO0']

        # #alternateacc
        # api_keys4 = ['AIzaSyD5CILfjYVfkrLIt9BK3eoXomLBOe-bO_k',
        #         'AIzaSyAlVpk9IWOxgLE9RYOmAR4KPg85gldEaBs',
        #         'AIzaSyDWQ01DT8_x-Jxpw3tvrvRtYJM9F-JyJUg',
        #         'AIzaSyAEeoSJI9Nytb3alK5_PKUbo041RC-GjSU',
        #         'AIzaSyDa2c3Xt3IqllAOOScIi5_1xAViWNMVcus',
        #         'AIzaSyAsz0czIheKaQGCIUc2IHwyVFklXHNeyw0',
        #         'AIzaSyAbQDpL_TyvGFwwi6joabbgj0PAPsrDFHY',
        #         'AIzaSyAUdPhKJZoV6YbJb53j3D3HhBE2NszT-y0',
        #         'AIzaSyC7BkkZtsk_pEJOj1nKFGf0pr7uVehLOr4',
        #         'AIzaSyBPdm_5aol47Wie0ppxj8SJEUGQe5Wur5I'
        #         ]
        
        task_keys = api_keys[args.task_description]
    else:
        client = None
        
    # # llava deepspeed
    # if 'llava-onevision-72b' in engine:
    #     from accelerate import Accelerator, DistributedType
    #     from accelerate.utils import set_seed
    #     import deepspeed
    #     from transformers.integrations.deepspeed import (
    #         deepspeed_init,
    #         deepspeed_load_checkpoint,
    #     )
        
    #     torch.distributed.init_process_group(backend="nccl")
    #     set_seed(args.run)
    #     accelerator = Accelerator(gradient_accumulation_steps=1)
    #     accelerator.wait_for_everyone()
        
    #     from datasets import load_dataset, Dataset
    #     from torch.utils.data import DataLoader
    #     # dataset = load_dataset("json", data_files="./new_data/intervention/pendulum/query.json")
    #     data = np.random.rand(16)
    #     label = np.random.randint(0, 2, size=16)
    #     ds = Dataset.from_dict({"data": data, "label": label}).with_format("torch")
    #     dataloader = DataLoader(ds, batch_size=4)
        
    #     for n,p in model.named_parameters():
    #         if p.requires_grad and 'projector' not in n:
    #             p.requires_grad = False
    #     # exit(0)
        
    #     grouped_params = [
    #         {
    #             "params": [
    #                 p for n, p in model.named_parameters() if 'projector' in n
    #             ],
    #             "weight_decay": 0.01
    #         },
    #     ]
    #     print(grouped_params)
    #     # model.half().eval()
    #     optimizer = torch.optim.AdamW(grouped_params, lr=1e-4)
    #     _, dummy_loader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    #     del dummy_loader, optimizer, dataloader
    #     print('After accelerator.prepare')
        
    #     model = e_prepare_deepspeed(model, accelerator)
    #     progress_bar = tqdm(len(query_meta), disable=not accelerator.is_local_main_process)

    #     for i, query in enumerate(query_meta):
    #         with torch.no_grad():
    #             predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
    #                                                 n_shot_support, data_path, processor, max_new_tokens, client=client, use_cot=args.cot)
    #             progress_bar.update(1)
    #     query['prediction'] = predicted_answer
    #     print(predicted_answer)
    #     results.append(query)
        
    #     exit(0)
                
    
    for i, query in enumerate(tqdm(query_meta)):
        
        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, query=query, selection_strategy=args.strategy)
        
        if 'gemini' in engine:
            temp = []
            api_key = task_keys[f'{args.dataset}_{n_shot}'][0]
            
            
            if i % 499 == 0:
                task_keys = api_keys[args.task_description + '_replacement']
                api_key = task_keys[f'{args.dataset}_{n_shot}'][0]
                
            
            client = genai.Client(api_key=api_key)
            
            if '2_5' in engine:
                if i > 0 and i%9 == 0:
                    time.sleep(61)
            else:
                if i > 0 and i%14 == 0:
                    time.sleep(61)
                
                
            # if n_shot == 8 and i==500 and ('circuit' in args.dataset):
            #     time.sleep(61)
            #     api_key = task_keys[f'{args.dataset}_{nshot}'][1]

        try:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                        n_shot_support, data_path, processor, max_new_tokens, client=client, use_cot=args.cot, idx=i)
            query['prediction'] = predicted_answer
            # print(predicted_answer)
            # exit(0)
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