#!/bin/bash
. environment.sh
conda activate qwen
python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 200 --cot True --samples 100
python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 200 --cot True --samples 100
python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 3 --task_description discovery --dataset pendulum --max_new_tokens 200 --cot True
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 4 --task_description discovery --dataset pendulum --max_new_tokens 200 --cot True
wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description discovery --dataset flow --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description discovery --dataset flow --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description discovery --dataset flow --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 3 --task_description discovery --dataset flow --max_new_tokens 200 --cot True
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 4 --task_description discovery --dataset flow --max_new_tokens 200 --cot True
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description discovery --dataset circuit --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description discovery --dataset circuit --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description discovery --dataset circuit --max_new_tokens 200 --cot True --samples 100
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 3 --task_description discovery --dataset circuit --max_new_tokens 200 --cot True
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 4 --task_description discovery --dataset circuit --max_new_tokens 200 --cot True
# wait


