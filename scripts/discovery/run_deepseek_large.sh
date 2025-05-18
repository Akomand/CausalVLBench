#!/bin/bash
. environment.sh
conda activate deepseek

# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 300 --cot True --strategy random
# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 300 --cot True --strategy random
# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 300 --cot True --strategy random

# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description discovery --dataset flow --max_new_tokens 300 --cot True --strategy random
# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 1 --task_description discovery --dataset flow --max_new_tokens 300 --cot True --strategy random
# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 2 --task_description discovery --dataset flow --max_new_tokens 300 --cot True --strategy random

python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 0 --task_description discovery --dataset circuit --max_new_tokens 300 --cot True --strategy random
# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 1 --task_description discovery --dataset circuit --max_new_tokens 300 --cot True --strategy random
# python I2T_inference.py --engine deepseek-vl2-large --n_shot 0 --run 2 --task_description discovery --dataset circuit --max_new_tokens 300 --cot True --strategy random
