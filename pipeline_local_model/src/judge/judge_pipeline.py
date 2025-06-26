import sys

from .load_config import load_judge_config

from .judge_cce import cce
from .judge_naive import naive

def judge_call(reward_data, judge_strategy, llm_models, reward_benchmark):
    judge_config = load_judge_config(judge_strategy, reward_benchmark)
    print(llm_models)
    if judge_strategy == "cce":
        return cce(reward_data, llm_models, judge_config, reward_benchmark)
    elif judge_strategy == "naive":
        return naive(reward_data, llm_models, judge_config, reward_benchmark)