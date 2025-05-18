#!/bin/bash
. environment.sh
conda activate deepseek
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 1 

CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 0 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 0 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 0 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 0 --task_description discovery --dataset flow --max_new_tokens 1

CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 0 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 0 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 0 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 0 --task_description discovery --dataset circuit --max_new_tokens 1


CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 1 
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 1 



CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 1 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 1 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 1 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 1 --task_description discovery --dataset flow --max_new_tokens 1

CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 1 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 1 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 1 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 1 --task_description discovery --dataset circuit --max_new_tokens 1


CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 1 

CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 2 --task_description discovery --dataset flow --max_new_tokens 1 
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 2 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 2 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 2 --task_description discovery --dataset flow --max_new_tokens 1

CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 2 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 2 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 2 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 2 --task_description discovery --dataset circuit --max_new_tokens 1


CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 3 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 3 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 3 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 3 --task_description discovery --dataset pendulum --max_new_tokens 1

CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 3 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 3 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 3 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 3 --task_description discovery --dataset flow --max_new_tokens 1

CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 3 --task_description discovery --dataset circuit --max_new_tokens 1 
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 3 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 3 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 3 --task_description discovery --dataset circuit --max_new_tokens 1


CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 4 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 4 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 4 --task_description discovery --dataset pendulum --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 4 --task_description discovery --dataset pendulum --max_new_tokens 1

CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 4 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 4 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 4 --task_description discovery --dataset flow --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 4 --task_description discovery --dataset flow --max_new_tokens 1

CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 0 --run 4 --task_description discovery --dataset circuit --max_new_tokens 1 
# CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine deepseek-vl2 --n_shot 2 --run 4 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine deepseek-vl2 --n_shot 4 --run 4 --task_description discovery --dataset circuit --max_new_tokens 1 &
# CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine deepseek-vl2 --n_shot 8 --run 4 --task_description discovery --dataset circuit --max_new_tokens 1
