from .openai_api import thread_evaluate_api, thread_generate_api
from .local_model import local_evaluate_vllm, local_generate_vllm
from .utils import *
from .load_model import IS_API
def retrieve_judgment(target_filepath, pairs_prompts):
    judgments = []
    for prompt in pairs_prompts:
        for i in range(50):
            #datapath = data_root+"/buffer/"+"thread_output_buffer/"+evaluate_prompt+"generate_"+model_name+"_OUTPUT*"
            for case in read_jsonl(target_filepath.replace("*", "_thread"+str(i))+".jsonl"):
                print(case[:-1])
                print("---------------------")
                print(prompt)
                exit()
                if prompt == case[:-1]:
                    prompt.append(case[-1])
                    judgments.append(prompt)
                    break
    return judgments

def call_judge_infer(judge_model, pairs_prompts, config):
    #exit()
    if judge_model.model_name in IS_API:
        target_filepath = thread_evaluate_api(pairs_prompts, judge_model)
    else:
        local_evaluate_vllm(pairs_prompts, judge_model, config)

    #judgments = retrieve_judgment(target_filepath, pairs_prompts)
    return pairs_prompts

def call_generate_infer(generation_model, instructions, config):
    if generation_model.model_name in IS_API:
        target_filepath = thread_generate_api(instructions, generation_model)
        return [ins[-1] for ins in instructions]
    else:
        return local_generate_vllm(instructions, generation_model, config)
    #print(pairs_prompts[0])
    #exit()
    #judgments = retrieve_judgment(target_filepath, pairs_prompts)
    
