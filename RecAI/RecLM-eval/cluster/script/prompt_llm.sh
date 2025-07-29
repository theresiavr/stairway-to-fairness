#!/bin/bash
#SBATCH --ntasks=1 --mem=20000M
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --time=6:00:00
#SBATCH --mail-type=START,END,FAIL

echo prompt_llm.sh
hostname
nvidia-smi

echo $CUDA_VISIBLE_DEVICES

py=/home/user_id/anaconda3/envs/intersect/bin/python3.11
eval=/home/user_id/intersectional-fairness/RecAI/RecLM-eval/eval.py

cd /home/user_id/intersectional-fairness/RecAI/RecLM-eval/

task=retrieval
v_num=1

models=("mistralai/Ministral-8B-Instruct-2410" "Qwen/Qwen2.5-7b-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "THUDM/glm-4-9b-chat")
for model in "${models[@]}"
    do
        data=ml-1m

        echo "Running task: $model for $data - neutral"
        $py $eval --task-names $task \
            --bench-name $data \
            --model_path_or_name $model \
            --batch_size 256 \
            --version_num $v_num \
            --prompt_type neutral 

        echo "Running task: $model for $data - sensitive"
        $py $eval --task-names $task \
            --bench-name $data \
            --model_path_or_name $model \
            --batch_size 256 \
            --version_num $v_num \
            --prompt_type sensitive 

        data=jobrec

        echo "Running task: $model for $data - neutral"
        $py $eval --task-names $task \
            --bench-name $data \
            --model_path_or_name $model \
            --batch_size 256 \
            --version_num $v_num \
            --prompt_type neutral 

        echo "Running task: $model for $data - sensitive"
        $py $eval --task-names $task \
            --bench-name $data \
            --model_path_or_name $model \
            --batch_size 256 \
            --version_num $v_num \
            --prompt_type sensitive 
    done