#!/bin/bash
#SBATCH --ntasks=1 --mem=12000M
#SBATCH --time=1:00:00
#SBATCH --mail-type=START,END,FAIL

echo generate_prompt.sh
hostname
nvidia-smi

echo $CUDA_VISIBLE_DEVICES

py=/home/user_id/anaconda3/envs/intersect/bin/python3.11
prg=/home/user_id/stairway-to-fairness/RecAI/RecLM-eval/preprocess/generate_data.py

cd /home/user_id/stairway-to-fairness/RecAI/RecLM-eval/

num_sample=20000
task_type=retrieval
v_num=1

# set up data
data=ml-1m
$py $prg --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type neutral
$py $prg --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type sensitive

data=lfm-1b
$py $prg --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type neutral
$py $prg --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type sensitive

data=jobrec
$py $prg --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type neutral
$py $prg --tasks $task_type --sample_num $num_sample --dataset $data --version_num $v_num --prompt_type sensitive


