import sys
sys.path.append("")
import argparse
import json
import os
import vllm
import torch

from eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model, dynamic_import_function

def generate(instances, llm_model_config, chat_formatting_function="eval.templates.create_prompt_with_tulu_chat_format", use_chat_format=True, use_vllm=True):
    prompts = []
    chat_formatting_function = dynamic_import_function(chat_formatting_function) if use_chat_format else None
    for instance in instances:
        if "messages" in instance:
            if not use_chat_format:
                raise ValueError("If `messages` is in the instance, `use_chat_format` should be True.")
            assert all("role" in message and "content" in message for message in instance["messages"]), \
                "Each message should have a `role` and a `content` field."
            #prompt = eval(args.chat_formatting_function)(instance["messages"], add_bos=False)
            prompt = chat_formatting_function(instance["messages"], add_bos=False)
        elif "prompt" in instance:
            if use_chat_format:
                messages = [{"role": "user", "content": instance["prompt"]}]
                prompt = chat_formatting_function(messages, add_bos=False)
            else:
                prompt = instance["prompt"]
        else:
            raise ValueError("Either `messages` or `prompt` should be in the instance.")
        prompts.append(prompt)
    if use_vllm:
        model = vllm.LLM(
            model=llm_model_config["model_path"],
            tokenizer=llm_model_config["model_path"],
            tokenizer_mode="slow" if llm_model_config["use_slow_tokenizer"] else "auto",
            tensor_parallel_size=torch.cuda.device_count(),
        )
        sampling_params = vllm.SamplingParams(
            temperature=llm_model_config["temperature"] if llm_model_config["do_sample"] else 0, 
            top_p=llm_model_config["top_p"],
            max_tokens=llm_model_config["max_new_tokens"],
        )
        outputs = model.generate(prompts, sampling_params)
        outputs = [it.outputs[0].text for it in outputs]
    else:
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=llm_model_config["model_name_or_path"], 
            tokenizer_name_or_path=llm_model_config["model_name_or_path"],
            load_in_8bit=llm_model_config["load_in_8bit"], 
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=llm_model_config["gptq"],
            use_fast_tokenizer=not llm_model_config["use_slow_tokenizer"],
        )
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=1,
            max_new_tokens=llm_model_config["max_new_tokens"],
            do_sample=llm_model_config["do_sample"],
            temperature=llm_model_config["temperature"],
            top_p=llm_model_config["top_p"],
        )
    del model
    print("Done.")
    return outputs