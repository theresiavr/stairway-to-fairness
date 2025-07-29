import os
import logging

from evaluates.evaluate import custom_compute_metrics_on_title_recommend
from utils import *

from glob import glob

import pickle

allow_regenerate = os.getenv("ALLOW_REGENERATE", "False").lower() == "true"

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

output_folder = "RecAI/RecLM-eval/output"

task_name = "retrieval"
list_data = ["ml-1m", "jobrec", "lfm-1b"]

for data in list_data:
    meta_data_file = f"RecAI/RecLM-eval/data/{data}/metadata.json"

    cand_answer_files = glob(f"{output_folder}/{data}/*/*")

    for f in cand_answer_files:
        #only compute on cleanded file
        if "cleaned" not in f:
            continue

        logger.info(f"Doing {f}")
        answer_file = f

        result, per_user_result = custom_compute_metrics_on_title_recommend(answer_file, meta_data_file)

        logger.info("Finish computing measures")

        the_path, model, file = f.split("\\")
        model = model.split("_")[0]
        prompt_type = file.split("_")[2]

        with open(f"results_llm/{data}_{model}_{prompt_type}_result.pickle","wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

        with open(f"results_llm/{data}_{model}_{prompt_type}_per-user-result.pickle","wb") as f:
            pickle.dump(per_user_result, f, pickle.HIGHEST_PROTOCOL)