import os
import fire
import copy

from src.utils import *
from src.data import load_instructions_data, load_pairs_data
from src.load_model import load_generation_model1, load_generation_model2, load_judge_model
from src.infer import call_generate_infer, call_judge_infer
from src.meta_metric import compute_acc, compute_corr
from src.prompt import generate_judge_prompts, generate_response_prompts
config_root = ".\configs"
data_root = ".\datafolder"
#* Configuration Part
def load_config(evaluation_data, judgment_prompt, generation_llm, judge_llm, overwrite_generate, overwrite_judge, w_label):
    config = read_json(os.path.join(config_root, evaluation_data+".json"))
    if generation_llm is not None and "," in generation_llm:
        config["generation_llm1"] = read_json(os.path.join(config_root, "generation_llm", generation_llm.split(",")[0]+'.json')) if generation_llm.split(",")[0] != "default" else "default"
        config["generation_llm2"] = read_json(os.path.join(config_root, "generation_llm", generation_llm.split(",")[1]+'.json')) if generation_llm.split(",")[1] != "default" else "default"
    elif generation_llm is not None:
        config["generation_llm"] = read_json(os.path.join(config_root, "generation_llm", generation_llm+'.json'))
    config["judge_llm"] = read_json(os.path.join(config_root, "judge_llm", judge_llm+'.json'))
    config["judgment_prompt"] = judgment_prompt
    config["overwrite_generate"] = overwrite_generate
    config["overwrite_judge"] = overwrite_judge
    config["w_label"] = w_label

    return config

#* Generation Part
def generate_w_llm(instructions, config):
    model1 = load_generation_model1(config)
    model2 = load_generation_model2(config)
    instructions_prompts = generate_response_prompts(config, instructions)
    instructions_prompts_1 = copy.copy(instructions_prompts)
    instructions_prompts_2 = copy.copy(instructions_prompts)
    responses_1 = call_generate_infer(model1, instructions_prompts_1, config) 
    responses_2 = call_generate_infer(model2, [_[:-1] for _ in instructions_prompts_2], config) 
    print(len(instructions), len(responses_1), len(responses_2))
    write_response1, write_response2 = [], []
    for instructionid, pair in enumerate(instructions):
        instructions[instructionid]["answer_a"] = [{"content":instructions_prompts[instructionid][1]}, {"content":responses_1[instructionid]}]
        instructions[instructionid]["answer_b"] = [{"content":instructions_prompts[instructionid][1]}, {"content":responses_2[instructionid]}]
        write_response1.append(responses_1[instructionid])
        write_response2.append(responses_2[instructionid])
    return instructions, write_response1, write_response2

#* Evaluation Part
def evaluate_w_llm_judge(pairs, config):
    judge_model = load_judge_model(config)
    judge_prompts = generate_judge_prompts(config, pairs)
    judgments = call_judge_infer(judge_model, judge_prompts, config)
    print(len(pairs), len(judgments))
    for pairid, pair in enumerate(pairs):
        pairs[pairid]["judgment_output"] = judgments[pairid][-1]
    return pairs

#* Meta-Evaluation Part
def meta_evaluate_w_metric(judgments, config):
    if config["evaluate_mode"] == "preference":
        compute_acc(judgments)
    elif config["evaluate_mode"] == "single_score":
        compute_corr(judgments)

#* Main-Function Part
def main(evaluation_data, judgment_prompt, generation_llm, judge_llm, overwrite_generate, overwrite_judge, w_label):
    config = load_config(evaluation_data, judgment_prompt, generation_llm, judge_llm, overwrite_generate, overwrite_judge, w_label)
    if overwrite_generate:
        instructions = load_instructions_data(config)
        pairs, write_response1, write_response2 = generate_w_llm(instructions, config)
        #write_output("E:\self_consistency_judge\/results/"+"_".join([str(evaluation_data), str(judgment_prompt), str(generation_llm), str("None")])+".json", pairs)
        write_output("E:\self_consistency_judge\/results/"+"_".join([str(evaluation_data), str(judgment_prompt), str(generation_llm.split(",")[0]), str("None")])+".json", write_response1)
        write_output("E:\self_consistency_judge\/results/"+"_".join([str(evaluation_data), str(judgment_prompt), str(generation_llm.split(",")[1]), str("None")])+".json", write_response2)
    else:
        pairs = load_pairs_data(config)
    if overwrite_judge:
        judgments = evaluate_w_llm_judge(pairs, config)
        write_output("E:\self_consistency_judge\/results/"+"_".join([str(evaluation_data), str(judgment_prompt), str(generation_llm), str(judge_llm)])+".json", judgments)
    else:
        judgments = read_json("E:\self_consistency_judge\/results/")
    #meta_evaluate_w_metric(judgments, config)

if __name__ == '__main__':
    fire.Fire(main)