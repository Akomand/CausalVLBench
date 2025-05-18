#!/bin/bash
. environment.sh
conda activate deepseek
python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description intervention --dataset pendulum --max_new_tokens 50
python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description intervention --dataset flow --max_new_tokens 50
python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description intervention --dataset circuit --max_new_tokens 50
