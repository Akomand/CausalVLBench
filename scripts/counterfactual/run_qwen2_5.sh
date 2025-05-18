#!/bin/bash
. environment.sh
conda activate qwen
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 40 --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 --cot True --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 --cot True --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 40 --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200 --cot True --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200 --cot True --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200 
# wait
python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 40 --strategy random
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200 --cot True
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200 --cot True
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# wait






# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 0 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 1 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 2 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 3 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 4 --task_description counterfactual --dataset pendulum --max_new_tokens 200 
# wait

# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 0 --task_description counterfactual --dataset flow --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 1 --task_description counterfactual --dataset flow --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 2 --task_description counterfactual --dataset flow --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 3 --task_description counterfactual --dataset flow --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 4 --task_description counterfactual --dataset flow --max_new_tokens 200
# wait


# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 0 --task_description counterfactual --dataset circuit --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 1 --task_description counterfactual --dataset circuit --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200  
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 2 --task_description counterfactual --dataset circuit --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 3 --task_description counterfactual --dataset circuit --max_new_tokens 200
# wait
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 0 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 2 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 4 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 200 
# python I2T_inference.py --engine qwen-vl-2.5-instruct --n_shot 8 --run 4 --task_description counterfactual --dataset circuit --max_new_tokens 200
# wait