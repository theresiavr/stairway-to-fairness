# ⚖️ Stairway to Fairness: Connecting Group and Individual Fairness (RecSys'25 Short Paper) 

This repository contains the appendix as well as the code used for the experiments and analysis in "Stairway to Fairness: Connecting Group and Individual Fairness" by Theresia Veronika Rampisela, Maria Maistro, Tuukka Ruotsalo, Falk Scholer, and Christina Lioma. This work has been accepted to RecSys 2025 as a short paper.

[[ACM] (not active yet)](https://doi.org/10.1145/3705328.3748031)

# Abstract
Fairness in recommender systems (RSs) is commonly categorised into group fairness and individual fairness. However, there is no established scientific understanding of the relationship between the two fairness types, as prior work on both types has used different evaluation measures or evaluation objectives for each fairness type, thereby not allowing for a proper comparison of the two. As a result, it is currently not known how increasing one type of fairness may affect the other. To fill this gap, we study the relationship of group and individual fairness through a comprehensive comparison of evaluation measures that can be used for both fairness types. Our experiments with 8 runs across 3 datasets show that recommendations that are highly fair for groups can be very unfair for individuals. Our finding is novel and useful for RS practitioners aiming to improve the fairness of their systems.

# Citation

```BibTeX
@inproceedings{Rampisela2025StairwayFairness,
author = {Rampisela, Theresia Veronika and Maistro, Maria and Ruotsalo, Tuukka and Scholer, Falk and Lioma, Christina},
title = {Stairway to Fairness: Connecting Group and Individual Fairness},
year = {2025},
isbn = {9798400713644},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3705328.3748031},
doi = {10.1145/3705328.3748031},
location = {Prague, Czech Republic},
series = {RecSys '25}
}
```

# License and Terms of Usage
The code is usable under the MIT License. Please note that RecAI (RecLM-eval) may have different terms of usage (see [their page](https://github.com/microsoft/RecAI/tree/main/RecLM-eval) for updated information).

# Datasets
## 1. Downloads
We use the **ML-1M**, **JobRec**, and **LFM-1B** datasets.

- The **ML-1M** and **LFM-1B** datasets can be downloaded from the Google Drive folder provided by [RecBole](https://recbole.io/dataset_list.html), under ProcessedDatasets:

    - ML-1M: Under MovieLens, download `ml-1m.zip`
    - LFM-1B: Go to LFM-1b > merged and download `lfm1b-artists.zip`.

- The **JobRec** dataset can be downloaded from the [Job Recommendation Challenge - Kaggle](https://www.kaggle.com/competitions/job-recommendation/data). This requires signing-in and accepting the competition rules.

1. Create a new folder `raw_data` and place the zip files in it.
2. Extract the zip files

## 2. Preprocessing

1. Create the folder `cleaned_data` and create the folders `ml-1m`, `jobrec`, and `lfm-1b` in it.
2. Go to the `preprocess` folder.
3. Run `ml1m.ipynb`, `job_rec.ipynb`, and `lfm.ipynb` to preprocess the datasets.

The preprocessed data will be saved in the `cleaned_data` folder.

## 3. Conversion
To convert the preprocessed/cleaned data to a data format that works for the next step (recommendation generation), do the following steps:

1. Go to RecAI/RecLM-eval and create a folder called `data`.
2. Run `preprocess/convert_to_recai.ipynb`

The formatted data will be saved in `RecAI/RecLM-eval/data`.

# LLMRecs: Generating Recommendations with Large Language Models (LLMs)

For this part of the work, we modified RecLM-eval from RecAI to suit our use case. The modified library is included in this repository in `RecAI/RecLM-eval`.

## 1. Prompt generation
This part generates prompt based on the templates in `RecAI/RecLM-eval/preprocess/{prompt_type}_{dataset}_templates.py`. There are two prompt types (neutral/non-sensitive and sensitive) per dataset. The sensitive prompt contains the user's sensitive attribute, while the non-sensitive prompt does not.

We used a Slurm cluster to generate the prompts for the three datasets:

```bash
cd RecAI/RecLM-eval/cluster/script
sbatch generate_prompt.sh
```

Alternatively, run the following:
```bash
cd RecAI/RecLM-eval/preprocess/

num_sample=20000
task_type=retrieval
v_num=1


data=ml-1m
python generate_data.py --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type neutral
python generate_data.py --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type sensitive

data=jobrec
python generate_data.py --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type neutral
python generate_data.py --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type sensitive

data=lfm-1b
python generate_data.py --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type neutral
python generate_data.py --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type sensitive

```

The generated prompts will be saved in .jsonl under `RecAI/RecLM-eval/data`.


## 2. Prompting the LLMs
We prompt Ministral-8B, Qwen2.5-7B, Llama-3.1-8B, and GLM-4-9B with the following scripts to generate recommendations per user:

```bash
cd RecAI/RecLM-eval/cluster/script

# prompt LLMs for ML-1M and JobRec
sbatch prompt_llm.sh

# prompt LLMs for LFM-1B
sbatch prompt_llm_big.sh
```

The LLM output can be found under `RecAI/RecLM-eval/output`.

Note that it may be necessary to request access to the gated models via HuggingFace, prior to running this part.

## 3. Post-processing the LLM recommendation output

To clean-up the LLM output, run the following:

```bash
cd RecAI/RecLM-eval
python cleaner.py
```

The cleaned output will be saved under `RecAI/RecLM-eval/output` and has "cleaned_retrieval" in the filename.

# LLMRecs Effectiveness and Fairness Evaluation

1. We evaluate recommendation effectiveness per user:

```bash
mkdir results_llm
cd RecAI/RecLM-eval
python eval_and_save.py
```

The results will be saved under `results_llm`.

2. Further analyses can be done by running the following notebooks in the `eval` folder:

- `eval_LLM.ipynb`: This computes the overall effectiveness and fairness scores, as well as runs the intersectional fairness and fairness decomposability analyses. Note that this notebook must be run first, before the other two.
- `corr_LLM.ipynb`: This generates the correlation heatmap between individual and group user fairness measures.
- `generate_teaser.ipynb`: This generates the boxplots in the teaser image.
