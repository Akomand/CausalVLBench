#!/bin/bash
. environment.sh
conda activate vlm

python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description discovery_interleaved --dataset pendulum --samples 100 --strategy random --cot True
python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description discovery_interleaved --dataset flow --samples 100 --strategy random --cot True
python I2T_inference.py --engine gemini --n_shot 0 --run 0 --task_description discovery_interleaved --dataset circuit --samples 100 --strategy random --cot True