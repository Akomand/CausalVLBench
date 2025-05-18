#!/bin/bash
. environment.sh
conda activate vlm

python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 2 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 4 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 8 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 50

python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 2 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 4 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 8 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 50

python I2T_inference.py --engine llava-onevision-72b --n_shot 0 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 2 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 4 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 50
python I2T_inference.py --engine llava-onevision-72b --n_shot 8 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 50
