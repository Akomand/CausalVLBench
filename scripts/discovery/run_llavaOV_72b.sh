#!/bin/bash
. environment.sh
conda activate vlm

# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 0 --task_description discovery --dataset pendulum --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 1 --task_description discovery --dataset pendulum --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 2 --task_description discovery --dataset pendulum --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 3 --task_description discovery --dataset pendulum --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 4 --task_description discovery --dataset pendulum --max_new_tokens 1
# wait
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 0 --task_description discovery --dataset flow --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 1 --task_description discovery --dataset flow --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 2 --task_description discovery --dataset flow --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 3 --task_description discovery --dataset flow --max_new_tokens 1
# python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 4 --task_description discovery --dataset flow --max_new_tokens 1
# wait
python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 0 --task_description discovery --dataset circuit --max_new_tokens 1
python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 1 --task_description discovery --dataset circuit --max_new_tokens 1
python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 2 --task_description discovery --dataset circuit --max_new_tokens 1
python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 3 --task_description discovery --dataset circuit --max_new_tokens 1
python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 4 --task_description discovery --dataset circuit --max_new_tokens 1