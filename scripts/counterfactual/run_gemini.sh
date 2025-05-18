#!/bin/bash
. environment.sh
conda activate vlm

# for i in {1..9}; do
#     python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --sub_samples $i
# done

# for i in {1..9}; do
#     python I2T_inference.py --engine gemini --n_shot 2 --run 0 --task_description counterfactual --dataset pendulum --sub_samples $i
# done

# for i in {1..9}; do
#     python I2T_inference.py --engine gemini --n_shot 4 --run 0 --task_description counterfactual --dataset pendulum --sub_samples $i
# done

# for i in {1..9}; do
#     python I2T_inference.py --engine gemini --n_shot 8 --run 0 --task_description counterfactual --dataset pendulum --sub_samples $i
# done

python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --cot True --strategy random 
# python I2T_inference.py --engine gemini --n_shot 2 --run 0 --task_description counterfactual --dataset pendulum  
# python I2T_inference.py --engine gemini --n_shot 4 --run 0 --task_description counterfactual --dataset pendulum  
# python I2T_inference.py --engine gemini --n_shot 8 --run 0 --task_description counterfactual --dataset pendulum  

# python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description counterfactual --dataset flow --cot True --strategy random 
# python I2T_inference.py --engine gemini --n_shot 2 --run 0 --task_description counterfactual --dataset flow  
# python I2T_inference.py --engine gemini --n_shot 4 --run 0 --task_description counterfactual --dataset flow  
# python I2T_inference.py --engine gemini --n_shot 8 --run 0 --task_description counterfactual --dataset flow 

# python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description counterfactual --dataset circuit --cot True --strategy random  
# python I2T_inference.py --engine gemini --n_shot 2 --run 0 --task_description counterfactual --dataset circuit  
# python I2T_inference.py --engine gemini --n_shot 4 --run 0 --task_description counterfactual --dataset circuit  
# python I2T_inference.py --engine gemini --n_shot 8 --run 0 --task_description counterfactual --dataset circuit 

