# CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models
This is the source code for the implementation of "CausalVLBench: Benchmarking Visual Causal
Reasoning in Large Vision-Language Models"

Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability. Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA). Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks. We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs. Our CausalVLBench encompasses three representative tasks: causal structure inference, intervention target prediction, and counterfactual prediction. We evaluate the ability of state-of-the-art open-source LVLMs on our causal reasoning tasks across three causal representation learning datasets and demonstrate their fundamental strengths and weaknesses. We hope that our benchmark elucidates the drawbacks of existing vision-language models and motivates new directions and paradigms in improving the visual causal reasoning abilities of LVLMs.

## Usage

### Training and evaluating 

1. Clone the repository

     ```
     git clone https://github.com/Akomand/CausalVLBench.git
     ```
2. Create environment and install dependencies
   ```
   conda env create -f requirements/requirements.txt
   ```
3. Generate synthetic datasets using scripts in ```data/data_generation/```
4. Create JSON files
   ```
   python eval_dataset.py
   ```
5. Run inference script
   ```
    ./scripts/[task]/run_[model].sh
   ```
6. Run evaluation script to obtain performance for all models
   ```
   python get_results.py
   ```

### Data acknowledgements
Experiments are run using adapted versions of the following datasets to evaluate our model:

#### Datasets
<details closed>
<summary>Pendulum Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>Water Flow Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>Causal Circuit Dataset</summary>

[Link to dataset](https://developer.qualcomm.com/software/ai-datasets/causalcircuit)
</details>
