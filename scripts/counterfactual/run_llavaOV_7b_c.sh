#!/bin/bash
. environment.sh
conda activate vlm
# python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 2 --run 0 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 4 --run 0 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 8 --run 0 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# wait
# python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 1 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 2 --run 1 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 4 --run 1 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 8 --run 1 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# wait
# python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 2 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 2 --run 2 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 4 --run 2 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 8 --run 2 --task_description counterfactual_nograph --dataset pendulum --max_new_tokens 50 
# wait

python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 --cot True
# python I2T_inference.py --engine llava-onevision-7b --n_shot 2 --run 0 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 4 --run 0 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 8 --run 0 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
wait
python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 --cot True
# python I2T_inference.py --engine llava-onevision-7b --n_shot 2 --run 1 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 4 --run 1 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 8 --run 1 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
wait
python I2T_inference.py --engine llava-onevision-7b --n_shot 0 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 --cot True
# python I2T_inference.py --engine llava-onevision-7b --n_shot 2 --run 2 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 4 --run 2 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine llava-onevision-7b --n_shot 8 --run 2 --task_description counterfactual_nograph --dataset circuit --max_new_tokens 200 
wait