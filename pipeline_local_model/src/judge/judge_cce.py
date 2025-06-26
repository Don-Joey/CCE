import os
import random
random.seed(42)
import time, openai, json
from ..grm.prediction_function import generate
from ..utils.io import *

OUTPUT_PATH = "/data/output/"
ROOT_PATH = ""

PRINCIPLES = {
    "rewardbench": "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response",
    "rewardbench-2": "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response"
}

RESPONSE_PROMPT = (
    "Instruction: {instruction}\n "
    "Response:"
)

JUDGE_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors {specific_principle}. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better. \n\n"
    "[Instruction]\n{instruction}\n[The Start of Assistant A's Answer]\n{output_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{output_2}\n[The End of Assistant B's Answer]"
)

CCE_JUDGE_PROMPT = """
Please act as an impartial and comprehensive judge evaluating the quality of two AI-generated responses (Assistant A and Assistant B) to the user's question. Your role is to determine which assistant better fulfills the user's needs according to multiple criteria, without focusing narrowly on any single aspect.

[Related Judgments - Internal Reference Only]: {previous_evaluation}.
These judgments are strictly internal; do not quote or refer to them. They are only to ensure you apply a consistent and comprehensive standard.

Evaluation Principles:
1. Holistic Assessment: Consider a wide range of factors—{specific_principle}—and maintain balance among these factors.
2. Comparative Contextualization: Although your final evaluation should be based solely on the two given responses, you may imagine or recall characteristics of other responses you've encountered in similar situations as background context. Implicitly inspire from the specific details in these related judgments for your current evaluation and refine the evaluation by identifying any missing dimensions or subpoints that are relevant for comparing Assistant A and Assistant B. Use this mental benchmark to ensure you apply consistent standards. Do not, however, introduce new information from outside sources into the final judgment. The goal is to remain aware of what an ideal response might look like, to ensure neither given answer is judged too narrowly.
3. Fairness and Impartiality: Do not be influenced by length, style, order of appearance, or assistant names. Evaluate purely on content quality.
4. Avoid External Biases: Rely only on the two given responses. Do not bring in outside data.

Instructions:
1. Please read the above related judgments carefully. Do not include the content under [Related Judgments] in your final output, nor should you explicitly reference them. These related judgments are for your internal reasoning only, serving as a mental benchmark to guide a comprehensive and fair evaluation. Use them as reference points to identify helpful aspects, potential weaknesses, or key criteria missed in prior assessments. Do not mention or hint at their existence in the final answer.
2. Evaluate both Assistant A’s and Assistant B’s answers thoroughly using the criteria above.
3. Compare their strengths and weaknesses, guided by your internal standard. Again, do not mention or allude to the internal references.
4. Begin your evaluation by comparing the two responses and provide a explanation.
5. End with your final verdict on which assistant performed better using the exact format:
   - [[A]] if Assistant A is better.
   - [[B]] if Assistant B is better.


[User Question]
{instruction}

[The Start of Assistant A's Answer]
{output_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{output_2}
[The End of Assistant B's Answer]
"""

def check_whether_crowd_responses(reward_benchmark, llm_model_config, crowd_assistant):
    if os.path.exists(OUTPUT_PATH+reward_benchmark+"_cce_"+llm_model_config["model_name"]+"_"+crowd_assistant+".json"):
        content = read_json(OUTPUT_PATH+reward_benchmark+"_cce_"+llm_model_config["model_name"]+"_"+crowd_assistant+".json")
    else:
        return False
    for case in content:
        if "cce_crowd_responses" not in case:
            return False
        if crowd_assistant not in case["cce_crowd_responses"]:
            return False
    return True

def check_whether_crowd_judgments(reward_benchmark, llm_model_config, crowd_assistant):
    content = read_json(OUTPUT_PATH+reward_benchmark+"_cce_"+llm_model_config["model_name"]+"_"+crowd_assistant+".json")
    for case in content:
        if "cce_crowd_judgments" not in case:
            return False
        if "judgment_a" not in case["cce_crowd_judgments"][crowd_assistant] or "judgment_b" not in case["cce_crowd_judgments"][crowd_assistant]:
            return False
    return True

def create_crowd_response_prompts(reward_data):
    prompts = []
    caseid = 0
    for case in reward_data:
        user_prompt = RESPONSE_PROMPT.format(
            instruction=case["instruction"],
        )
        prompts.append(
            {
                "id": str(caseid),
                "messages":[{"role": "user", "content": user_prompt}],
            }
        )
        caseid+=1
    return prompts


def create_crowd_judgment_prompts(reward_data, judge_config, crowd_assistant):
    revise_principle = PRINCIPLES[judge_config["criteria"]]
    prompts = []
    for caseid, case in enumerate(reward_data):
        user_prompt = JUDGE_PROMPT.format(
            instruction=case["instruction"],
            output_1=case["response_a"],
            output_2=case["cce_crowd_responses"][crowd_assistant]["response"],
            specific_principle=revise_principle,
        )
        prompts.append(
            {
                "id": str(caseid),
                "messages":[{"role": "user", "content": user_prompt}],
            }
        )
        user_prompt = JUDGE_PROMPT.format(
            instruction=case["instruction"],
            output_1=case["cce_crowd_responses"][crowd_assistant]["response"],
            output_2=case["response_b"],
            specific_principle=revise_principle,
        )
        prompts.append(
            {
                "id": str(caseid),
                "messages":[{"role": "user", "content": user_prompt}],
            }
        )

    return prompts

def gather_crowd(reward_data, reward_benchmark, crowd_assistants_list):
    for crowd_assistant in crowd_assistants_list:
        content = read_json(OUTPUT_PATH+reward_benchmark+"_cce_"+llm_model_config["model_name"]+"_"+crowd_assistant+".json")
        for instance, target in zip(reward_data, content):
            if "cce" not in instance:
                instance["cce"] = {
                    crowd_assistant: {
                        "response": target["cce_crowd_responses"][crowd_assistant]["response"],
                        "judgment_a": target["cce_crowd_judgments"][crowd_assistant]["judgment_a"],
                        "judgment_b": target["cce_crowd_judgments"][crowd_assistant]["judgment_b"]
                    }
                }
            else:
                instance["cce"][crowd_assistant] = {
                    "response": target["cce_crowd_responses"][crowd_assistant]["response"],
                    "judgment_a": target["cce_crowd_judgments"][crowd_assistant]["judgment_a"],
                    "judgment_b": target["cce_crowd_judgments"][crowd_assistant]["judgment_b"]
                }
    return reward_data

def critique_select(reward_data):
    CROWD_RESPONSES_SYMBOL = ["CDEFGHIJKLMNOPQRSTUVWXYZ"]
    instance_target_context = []
    for instance in reward_data:
        instance_crowd_judgments_selected_pool = []
        count = 0
        for key_1, value_1 in instance["cce"].items():
            symbol = CROWD_RESPONSES_SYMBOL[count]
            if "[[B]]" in value1["judgment_a"]:
                instance_crowd_judgments_selected_pool.append([symbol, value1["judgment_a"].replace("Assistant B", "Assistant "+symbol).replace("[[B]]", "[["+symbol+"]]").replace("Response B", "Assistant "+symbol)])
            if "[[A]]" in value1["judgment_b"]:
                instance_crowd_judgments_selected_pool.append([symbol, value1["judgment_b"].replace("Assistant A", "Assistant "+symbol).replace("[[A]]", "[["+symbol+"]]").replace("Response A", "Assistant "+symbol)])
        instance_target_context.append(instance_crowd_judgments_selected_pool)
    return instance_target_context
def removal_outcome(instance_target_context):
    for instance_pool_id, instance_pool in enumerate(instance_target_context):
        for instance_id in range(len(instance_pool)):
            instance_target_context[instance_pool_id][instance_id][1] = instance_pool[instance_id][1].replace("[[A]]", "").replace("[[B]]", "")
    return instance_target_context
def create_judgment_prompts(reward_data, instance_target_context, judge_config):
    judge_principle = PRINCIPLES[judge_config["criteria"]]
    prompts = []
    caseid = 0
    for case, case_crowd_context in zip(reward_data, instance_target_context):
        case_crowd_context = random.shuffle(case_crowd_context)[:judge_config["computation"]]
        context_string = ""
        COUNT = 1
        for crowd_jud in case_crowd_context:
            if "Assistant A" in crowd_jud[1] or "Response A" in crowd_jud[1]:
                context_string+=str(COUNT)+". Evaluation between Assistant A and Assistant "+crowd_jud[0]+": "+crowd_jud[1]+"\n"
            elif "Assistant B" in crowd_jud[1] or "Response B" in crowd_jud[1]:
                context_string+=str(COUNT)+". Evaluation between Assistant B and Assistant "+crowd_jud[0]+": "+crowd_jud[1]+"\n"
        user_prompt = CCE_JUDGE_PROMPT.format(
            instruction=case["instruction"],
            output_1=case["response_a"],
            output_2=case["response_b"],
            specific_principle=judge_principle,
            previous_evaluation=context_string
        )
        prompts.append(
            {
                "id": str(caseid),
                "messages":[{"role": "user", "content": user_prompt}],
            }
        )
        caseid+=1
    return prompts


def cce(reward_data, llm_model_config, judge_config, reward_benchmark):
    # generate crowd responses
    crowd_assistants_list = ["gemma3-1b-it", "gemma3-4b-it", "llama3.1-8b-instruct", "mistral-nemo-instruct-2407", "mistral0.3-7b-instruct", "qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct"]
    if judge_config["candidate_assistants"][1] - judge_config["candidate_assistants"][0] < len(crowd_assistants_list):
        for crowd_assistant in crowd_assistants_list[judge_config["candidate_assistants"][0]: judge_config["candidate_assistants"][1]]:
            if check_whether_crowd_responses(reward_benchmark, llm_model_config, crowd_assistant):
                pass
            else:
                assistant_config = read_json(os.path.join(ROOT_PATH, "data/configs", "assistant_llm", crowd_assistant+'.json'))
                crowd_responses_prompts = create_crowd_response_prompts(reward_data)
                crowd_responses_generation = generate(crowd_responses_prompts, assistant_config)
                for source, target in zip(reward_data, crowd_responses_generation):
                    source["cce_crowd_responses"] = {assistant_config["model_name"]: {"response": target}}
                write_output(OUTPUT_PATH+reward_benchmark+"_cce_"+llm_model_config["model_name"]+"_"+crowd_assistant+".json", reward_data)
            # generate crowd judgments
            if check_whether_crowd_judgments(reward_benchmark, llm_model_config, crowd_assistant):
                pass
            else:
                crowd_judgments_prompts = create_crowd_judgment_prompts(reward_data, judge_config, crowd_assistant)
                crowd_judgments_generation = generate(crowd_judgments_prompts, llm_model_config)
                for sid, source in enumerate(reward_data):
                    source["cce_crowd_judgments"] = {assistant_config["model_name"]: {"judgment_a": crowd_judgments_generation[sid]}}
                    source["cce_crowd_judgments"][assistant_config["model_name"]]["judgment_b"] = crowd_judgments_generation[sid+1]
                write_output(OUTPUT_PATH+reward_benchmark+"_cce_"+llm_model_config["model_name"]+"_"+crowd_assistant+".json", reward_data)

            print("Generate Crowd Done!")
            exit()
        
    else:
        # select & preprocess
        reward_data = gather_crowd(reward_data, reward_benchmark, crowd_assistants_list)
        instance_target_context = critique_select(reward_data)
        instance_target_context = removal_outcome(instance_target_context)

        # generate cce judgment
        judgment_prompts = create_judgment_prompts(reward_data, instance_target_context, judge_config)
        judgment_generation = generate(judgment_prompts, llm_model_config)
        for source, target in zip(reward_data, judgment_generation):
            source["cce_judgment"] = target
        return reward_data, "cce_judgment"

