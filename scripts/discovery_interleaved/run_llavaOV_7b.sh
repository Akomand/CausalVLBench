#!/bin/bash
. environment.sh
conda activate vlm
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description discovery_interleaved --dataset pendulum --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description discovery_interleaved --dataset flow --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description discovery_interleaved --dataset circuit --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 1 --task_description discovery_interleaved --dataset pendulum --max_new_tokens 1 &
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 1 --task_description discovery_interleaved --dataset flow --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 1 --task_description discovery_interleaved --dataset circuit --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 2 --task_description discovery_interleaved --dataset pendulum --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 2 --task_description discovery_interleaved --dataset flow --max_new_tokens 1 &
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 2 --task_description discovery_interleaved --dataset circuit --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 3 --task_description discovery_interleaved --dataset pendulum --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 3 --task_description discovery_interleaved --dataset flow --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=3 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 3 --task_description discovery_interleaved --dataset circuit --max_new_tokens 1 &
wait
CUDA_VISIBLE_DEVICES=0 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 4 --task_description discovery_interleaved --dataset pendulum --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=1 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 4 --task_description discovery_interleaved --dataset flow --max_new_tokens 1 &
CUDA_VISIBLE_DEVICES=2 python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 4 --task_description discovery_interleaved --dataset circuit --max_new_tokens 1 &
wait