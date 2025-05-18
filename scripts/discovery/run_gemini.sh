#!/bin/bash
. environment.sh
conda activate vlm

# python I2T_inference.py --engine gemini_2_5 --n_shot 0 --run 0 --task_description discovery --dataset pendulum --samples 100 --strategy random
# python I2T_inference.py --engine gemini_2_5 --n_shot 0 --run 0 --task_description discovery --dataset flow --samples 100 --strategy random
python I2T_inference.py --engine gemini_2_5 --n_shot 0 --run 0 --task_description discovery --dataset circuit --samples 100 --strategy random