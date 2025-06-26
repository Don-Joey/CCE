import os
import fire

from src.utils.io import *
from src.judge.judge_pipeline import judge_call
from src.evaluation.metrics import evaluate_reward
from src.utils.contant import *

OUTPUT_PATH = "/home/zhangqiyuan3/qiyuan/tts_for_grm/reviseval/data/output/"
def main(reward_benchmark="rewardbench", judge_strategy="cce", llm_models="qwen2.5-7b-instruct"):
    output_file_path = OUTPUT_PATH+REWARD_DATA[reward_benchmark].split("/")[-1][:-5]+"_"+judge_strategy+"_"+llm_models+".json"
    # read llm configuration
    llm_models = read_json(os.path.join(ROOT_PATH, "data/configs", "judge_llm", llm_models+'.json'))
    # read reward benchmarks
    reward_data = read_json(ROOT_PATH+REWARD_DATA[reward_benchmark])
    # run judge strategy
    reward_data, output_key = judge_call(reward_data, judge_strategy, llm_models, reward_benchmark)
    write_output(output_file_path, reward_data)
    # evaluate evaluation strategy
    evaluate_reward(reward_data, output_key)

if __name__ == '__main__':
    fire.Fire(main)