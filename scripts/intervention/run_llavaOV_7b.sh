#!/bin/bash
. environment.sh
conda activate vlm
python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description intervention --dataset pendulum  
python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description intervention --dataset flow  
python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description intervention --dataset circuit  