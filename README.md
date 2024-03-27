# Rectifying Demonstration Shortcut in In-Context Learning

This is the PyTorch implementation of **[Rectifying Demonstration Shortcut in In-Context Learning](https://arxiv.org/abs/2403.09488)**  (NAACL 2024 Main)

# Overview

Large language models (LLMs) are able to solve various tasks with only a few demonstrations utilizing their in-context learning (ICL) abilities. However, LLMs often rely on their pre-trained semantic priors of demonstrations rather than on the input-label relationships to proceed with ICL prediction. In this work, we term this phenomenon as the **'Demonstration Shortcut'.** While previous works have primarily focused on improving ICL prediction results for predefined tasks, we aim to rectify the Demonstration Shortcut, thereby enabling the LLM to effectively learn new input-label relationships from demonstrations. To achieve this, we introduce **In-Context Calibration**, a demonstration-aware calibration method. We evaluate the effectiveness of the proposed method in two settings: (1) the Original ICL Task using the standard label space and (2) the Task Learning setting, where the label space is replaced with semantically unrelated tokens. In both settings, In-Context Calibration demonstrates substantial improvements, with results generalized across three LLM families (OPT, GPT, and Llama2) under various configurations.

For the Task Learning Setting (where the label space is mapped to a string number), we recommend exchanging data_utils.py for data_utils_task_learning.py.

Our code is based on the code of [FewshotLearning](https://github.com/tonyzhaozh/few-shot-learning). Please refer to their repository for more detailed information.

## Pre-requisite

* torch
* bitsandbytes
* transformers
* peft
* accelerate
* openai
* pandas==2.0.2
* huggingface_hub
* scikit-learn==1.2.2 
* scipy==1.10.1 
* threadpoolctl==3.1.0

## Data

You can download the dataset from the following Google drive

> data: [data-drive](https://drive.google.com/drive/folders/1WnP1VVcFkNT5it6eyTF4IW3FNzlgKuDw?usp=sharing)

## Implementation

You can adjust the hyperparameters (e.g,.  $\lambda$) to suit your dataset environment. 

```console
sh shot_test.sh
```
