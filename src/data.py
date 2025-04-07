from datasets import load_dataset
import random
import os
from .utils import *

random.seed(42)
IN_DATASETS=["mtbench_human", "rewardbench", "helpsteer2", "judgebench", "evalbias"]

def unify_template(dataset, dataset_name):
    '''
    {
        "instruction": 
        "answer_a":
        "answer_b":
        "label":
        "multiturn":
    }
    {
        "instruction": 
        "answer":
        "label_score":
        "multiturn":
    }
    '''
    def from_mtbench_preference(ds):
        ds = ds["human"]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["conversation_a"][0]["content"]
            unified_format_case["answer_a"] = case["conversation_a"]
            unified_format_case["answer_b"] = case["conversation_b"]
            if case["winner"] == "model_a":
                unified_format_case["label"] = 1
            elif case["winner"] == "model_b":
                unified_format_case["label"] = 2
            else:
                unified_format_case["label"] = 3
            unified_format_case["multiturn"] = False if case["turn"] == 1 else True
            unified_format_case["external_info"] = (case["model_a"], case["model_b"])
            new_ds.append(unified_format_case)
        return new_ds
    def from_rewardbench_preference(ds):
        ds = ds["filtered"]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            if random.uniform(0, 1)<0.5:
                answer_a = [{"content": case["prompt"]}, {"content": case["chosen"]}]
                answer_b = [{"content": case["prompt"]}, {"content": case["rejected"]}]
                label = 1
                external_info = (case["chosen_model"], case["rejected_model"])
            else:
                answer_a = [{"content": case["prompt"]}, {"content": case["rejected"]}]
                answer_b = [{"content": case["prompt"]}, {"content": case["chosen"]}]
                label = 2
                external_info = (case["rejected_model"], case["chosen_model"])
            unified_format_case["instruction"] = case["prompt"]
            unified_format_case["answer_a"] = answer_a
            unified_format_case["answer_b"] = answer_b
            unified_format_case["label"] = label
            unified_format_case["multiturn"] = False
            unified_format_case["external_info"] = external_info
            new_ds.append(unified_format_case)
        return new_ds
    def from_helpsteer2_preference(ds):
        ds = ds["validation"]
        new_ds = []
        for cid, case in enumerate(ds):
            if cid %2 == 0:
                unified_format_case = {}
                unified_format_case["instruction"] = case["prompt"]
                unified_format_case["answer_a"] = [{"content": case["prompt"]}, {"content":case["response"]}]
                score_a = [0.65*case["helpfulness"], 0.8*case["correctness"], 0.45*case["coherence"], 0.55*case["complexity"], 0.4*-case["verbosity"]]
            elif cid%2 == 1:
                assert unified_format_case["instruction"] == case["prompt"], "不是一个case"
                unified_format_case["answer_b"] = [{"content": case["prompt"]}, {"content":case["response"]}]
                score_b = [0.65*case["helpfulness"], 0.8*case["correctness"], 0.45*case["coherence"], 0.55*case["complexity"], 0.4*-case["verbosity"]]
                unified_format_case["label"] = 2 if sum(score_a) < sum(score_b) else 1
                external_info = (score_a, score_b)
                unified_format_case["multiturn"] = False
                unified_format_case["external_info"] = external_info
                new_ds.append(unified_format_case)
        return new_ds
    def from_lfqa_preference(ds):
        new_ds = []
        for domain, domain_cases in ds.items():
            for case in domain_cases:
                unified_format_case = {}
                
                unified_format_case["instruction"] = case["question_text"]
                unified_format_case["answer_a"] = [{"content": case["question_text"]}, {"content":case["answer1"]}]
                unified_format_case["answer_b"] = [{"content": case["question_text"]}, {"content":case["answer2"]}]
                unified_format_case["label"] = 2 if case["BetterAnswer"][-1] == "B" else 1
                unified_format_case["multiturn"] = False
                unified_format_case["external_info"] = {
                    "domain": domain
                }
                new_ds.append(unified_format_case)
        return new_ds
    def from_judgebench_preference(ds):
        ds = [_ for _ in ds["gpt"]]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["question"]
            unified_format_case["answer_a"] = [{"content": case["question"]}, {"content":case["response_A"]}]
            unified_format_case["answer_b"] = [{"content": case["question"]}, {"content":case["response_B"]}]
            unified_format_case["label"] = 1 if "A>" in case["label"] else 2
            unified_format_case["multiturn"] = False
            unified_format_case["external_info"] = {
                "source": case["source"]
            }
            new_ds.append(unified_format_case)
        return new_ds
    def from_alpacaeval_preference(ds):
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["instruction"]
            unified_format_case["answer_a"] = [{"content": case["instruction"]}, {"content":case["output_1"]}]
            unified_format_case["answer_b"] = [{"content": case["instruction"]}, {"content":case["output_2"]}]
            unified_format_case["label"] = 1 if sum(case["preference"]) < 6.5 else 2
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    def from_evalbias_preference(ds):
        ds = [_ for _ in ds["train"]][:1000]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["instruction"]
            unified_format_case["answer_a"] = [{"content": case["instruction"]}, {"content":case["output_1"]}]
            unified_format_case["answer_b"] = [{"content": case["instruction"]}, {"content":case["output_2"]}]
            unified_format_case["label"] = case["label"]
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    def from_tulu3_preference(ds):
        ds = ds[10000:]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["prompt"]
            unified_format_case["answer_a"] = [{"content": case["prompt"]}, {"content":case["response_1"]}]
            unified_format_case["answer_b"] = [{"content": case["prompt"]}, {"content":case["response_2"]}]
            unified_format_case["label"] = case["label"]
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    if 'mtbench_human' == dataset_name:
        return from_mtbench_preference(dataset)
    elif 'rewardbench' == dataset_name:
        return from_rewardbench_preference(dataset)
    elif 'helpsteer2' == dataset_name:
        return from_helpsteer2_preference(dataset)
    elif 'lfqa' == dataset_name:
        return from_lfqa_preference(dataset)
    elif "judgebench" == dataset_name:
        return from_judgebench_preference(dataset)
    elif "alpacaeval" == dataset_name:
        return from_alpacaeval_preference(dataset)
    elif "evalbias" == dataset_name:
        return from_evalbias_preference(dataset)
    elif "tulu3" == dataset_name:
        return from_tulu3_preference(dataset)
    elif '...' in dataset_name:
        return None

def unify_generate_template(dataset, dataset_name):
    '''
    {
        "instruction": 
        "answer_a":
        "answer_b":
        "label":
        "multiturn":
    }
    {
        "instruction": 
        "answer":
        "label_score":
        "multiturn":
    }
    '''
    def from_mtbench_preference(ds):
        ds = ds["human"]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            if case["turn"] == 1:
                unified_format_case["instruction"] = case["conversation_a"][0]["content"]
                unified_format_case["multiturn"] = False #这里有问题
            else:
                unified_format_case["instruction"] = "<|user|>: "+case["conversation_a"][0]["content"]+"\n" + "<|assistant|>: "+case["conversation_a"][1]["content"]+"\n"
                unified_format_case["instruction"] += "<|user|>: "+case["conversation_a"][2]["content"]+"\n<|assistant|>: " 
                unified_format_case["multiturn"] = True
            #unified_format_case["default_answer_a"] = case["conversation_a"][1]["content"]
            #unified_format_case["default_answer_a"] = case["conversation_b"][1]["content"]
            new_ds.append(unified_format_case)
        return new_ds
    def from_rewardbench_preference(ds):
        ds = ds["filtered"]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["prompt"]
            unified_format_case["multiturn"] = False
            #print(case)
            #exit()
            #unified_format_case["default_answer_a"] = case["chosen"]
            #unified_format_case["default_answer_a"] = case["rejected"]
            new_ds.append(unified_format_case)
        return new_ds
    def from_helpsteer2_preference(ds):
        ds = ds["validation"]
        new_ds = []
        for cid, case in enumerate(ds):
            if cid %2 == 0:
                unified_format_case = {}
                unified_format_case["instruction"] = case["prompt"]
                unified_format_case["multiturn"] = False
                new_ds.append(unified_format_case)
        return new_ds
    def from_lfqa_preference(ds):
        new_ds = []
        for domain, domain_cases in ds.items():
            for case in domain_cases:
                unified_format_case = {}
                unified_format_case["instruction"] = case["question_text"]
                unified_format_case["multiturn"] = False
                new_ds.append(unified_format_case)
        return new_ds
    def from_judgebench_preference(ds):
        ds = [_ for _ in ds["gpt"]]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["question"]
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    def from_alpacaeval_preference(ds):
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["instruction"]
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    def from_evalbias_preference(ds):
        ds = [_ for _ in ds["train"]][:1000]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["instruction"]
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    def from_tulu3_preference(ds):
        ds = ds[10000:]
        new_ds = []
        for case in ds:
            unified_format_case = {}
            unified_format_case["instruction"] = case["prompt"]
            unified_format_case["multiturn"] = False
            new_ds.append(unified_format_case)
        return new_ds
    if 'mtbench_human' == dataset_name:
        return from_mtbench_preference(dataset)
    elif 'rewardbench' == dataset_name:
        return from_rewardbench_preference(dataset)
    elif 'helpsteer2' == dataset_name:
        return from_helpsteer2_preference(dataset)
    elif 'lfqa' == dataset_name:
        return from_lfqa_preference(dataset)
    elif 'judgebench' == dataset_name:
        return from_judgebench_preference(dataset)
    elif 'alpacaeval' == dataset_name:
        return from_alpacaeval_preference(dataset)
    elif 'evalbias' == dataset_name:
        return from_evalbias_preference(dataset)
    elif "tulu3" == dataset_name:
        return from_tulu3_preference(dataset)
    elif '...' in dataset_name:
        return None


def supplement_ds(ds, config, model, replace_flag):
    response_outputs = read_json(os.path.join(config["instructions_data_path"], "buffer/local_buffer/output", config["name"]+"_"+model["model_name"]+"_temperature_"+str(model["temperature"])+".json"))
    new_ds = []
    for caseid, case in enumerate(ds):
        if case["multiturn"]:
            if "judge_prompt" in response_outputs[caseid]:
                case["answer_"+replace_flag] = case["answer_"+replace_flag][:-1]+[{"content": response_outputs[caseid]['judge_prompt'][-1]}]
            else:
                case["answer_"+replace_flag] = case["answer_"+replace_flag][:-1]+[{"content": response_outputs[caseid]}]
        else:
            if "judge_prompt" in response_outputs[caseid]:
                case["answer_"+replace_flag] = [{"content": case["instruction"]}, {"content": response_outputs[caseid]['judge_prompt'][-1]}]
            else:
                case["answer_"+replace_flag] = [{"content": case["instruction"]}, {"content": response_outputs[caseid]}]
        new_ds.append(case)
    return new_ds

def load_instructions_data(config):
    if config["name"] in IN_DATASETS:
        ds = load_dataset(config["dataset_key"], cache_dir=config["instructions_data_path"])
    else:
        ds = read_json(config["instructions_data_path"])
    ds = unify_generate_template(ds, config["name"])
    return ds
def load_pairs_data(config):
    if config["name"] in IN_DATASETS:
        ds = load_dataset(config["dataset_key"], cache_dir=config["instructions_responses_pairs_data_path"])
    else:
        ds = read_json(config["instructions_responses_pairs_data_path"])
    ds = unify_template(ds, config["name"])
    if "generation_llm1" in config and "generation_llm2" in config:
        if config["generation_llm1"] != "default":
            ds = supplement_ds(ds, config, config["generation_llm1"], replace_flag="a")
        elif config["generation_llm2"] != "default":
            ds = supplement_ds(ds, config, config["generation_llm2"], replace_flag="b")
    return ds
