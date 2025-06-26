from src.utils.io import *
CONFIG_ROOT = "data/configs/"
def load_judge_config(judge_strategy, reward_benchmark):
    judge_config = read_json(CONFIG_ROOT+judge_strategy+".json")
    if judge_config["criteria"] == "default":
        judge_config["criteria"] = reward_benchmark
    return judge_config