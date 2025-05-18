#!/bin/bash
. environment.sh
conda activate vlm
python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 0 --task_description intervention --dataset pendulum --max_new_tokens 40
python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 0 --task_description intervention --dataset flow --max_new_tokens 40 
python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 0 --task_description intervention --dataset circuit --max_new_tokens 40 

