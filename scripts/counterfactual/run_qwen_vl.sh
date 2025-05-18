#!/bin/bash
. environment.sh
conda activate vlm
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 40 
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 40
wait

CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 40 
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 40
wait

CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 40 
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 40
wait

CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 40
wait

CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 40
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine qwen-vl-chat --n_shot 0 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine qwen-vl-chat --n_shot 2 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine qwen-vl-chat --n_shot 4 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 40 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine qwen-vl-chat --n_shot 8 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 40
wait