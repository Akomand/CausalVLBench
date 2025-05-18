#!/bin/bash
. environment.sh
conda activate deepseek

python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description discovery_interleaved --dataset pendulum --max_new_tokens 50
python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description discovery_interleaved --dataset flow --max_new_tokens 50
python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description discovery_interleaved --dataset circuit --max_new_tokens 50
