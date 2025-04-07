# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

#import uuid
#import asyncio
#import fire
import json
import os
from .utils import *
'''
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from accelerate.utils import is_xpu_available
from tqdm import tqdm
if is_xpu_available():
    torch.xpu.manual_seed(42)
else:
    torch.cuda.manual_seed(42)

torch.manual_seed(42)
'''
class Generation1_Local_configuration():
    def __init__(self, data_root, config):
        model_name = config["generation_llm1"]["model_name"]
        self.target = data_root+"/buffer/"+"thread_output_buffer/"+"default_generate_"+model_name+"_OUTPUT*"
        self.model_name = model_name
        self.temperature = config["generation_llm1"]["temperature"]

class Generation2_Local_configuration():
    def __init__(self, data_root, config):
        model_name = config["generation_llm2"]["model_name"]
        self.target = data_root+"/buffer/"+"thread_output_buffer/"+"default_generate_"+model_name+"_OUTPUT*"
        self.model_name = model_name
        self.temperature = config["generation_llm2"]["temperature"]


class Judge_Local_configuration():
    def __init__(self, data_root, config):
        evaluate_prompt = config["judgment_prompt"]
        model_name = config["judge_llm"]["model_name"]
        self.target = data_root+"/buffer/"+"thread_output_buffer/"+evaluate_prompt+"_evaluate_"+model_name+"_OUTPUT*"
        self.model_name = model_name
        self.temperature = config["judge_llm"]["temperature"]

def local_generate_vllm(instructions, generation_model, config):
    input_buffer_root = "E:\self_consistency_judge\datafolder\/buffer/"+"local_buffer/input"
    if os.path.isdir(input_buffer_root):
        pass
    else:
        os.mkdir(input_buffer_root)
    new_prompts=[]

    for prompt in instructions:
        new_prompts.append(prompt[1]["content"])
    write_output(input_buffer_root+"/"+config["name"]+".json", new_prompts)
    print("WRITE PROMPTS!")
    output_buffer_root = "E:\self_consistency_judge\datafolder\/buffer/"+"local_buffer/output"
    if os.path.isfile(output_buffer_root+"/"+config["name"]+"_"+generation_model.model_name+"_temperature_"+str(generation_model.temperature)+".json"):
        return read_json(output_buffer_root+"/"+config["name"]+"_"+generation_model.model_name+"_temperature_"+str(generation_model.temperature)+".json")
    else:
        exit()

def local_evaluate_vllm():
    pass

'''

def load_model(model_name, peft_model=None, pp_size=1, tp_size=1):
    additional_configs = {}
    if peft_model:
        additional_configs["enable_lora"] = True
        
    engine_config = AsyncEngineArgs(
        model=model_name,
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
        max_loras=1,
        **additional_configs)

    llm = AsyncLLMEngine.from_engine_args(engine_config)
    return llm

async def main(
    model,
    peft_model_name=None,
    max_new_tokens=1024,
    user_prompt=None,
    top_p=1.0,
    temperature=0.01
):
    #print(f"User prompt:\n{user_prompt}")

    print(f"sampling params: top_p {top_p} and temperature {temperature} for this inference request")
    sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens)

    lora_request = None
    if peft_model_name:
        lora_request = LoRARequest("lora",0,peft_model_name)

    req_id = str(uuid.uuid4())

    outputs = model.generate(user_prompt, sampling_param, req_id, lora_request=lora_request)
    output=None
    async for request_output in outputs:
        output = request_output
    return output.outputs[0].text
def run_script(
    model_name: str,
    peft_model_name=None,
    pp_size : int = 1,
    tp_size : int = 1,
    max_new_tokens=1024,
    user_prompt=None,
    top_p=1.0,
    temperature=0.01
):
    model = load_model(model_name, peft_model_name, pp_size, tp_size)
    final_outputs = []
    with open(user_prompt, "r") as f:
        user_prompts = json.load(f)            
    for k in tqdm(user_prompts):
        output = asyncio.get_event_loop().run_until_complete(main(model, peft_model_name, max_new_tokens, k, top_p, temperature))
        final_outputs.append(output)
    with open(user_prompt[:-5]+"_output.json", "w") as f:
        json.dump(final_outputs, f, indent=4)
     

if __name__ == "__main__":
    fire.Fire(run_script)
'''