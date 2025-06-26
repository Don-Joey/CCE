import random
random.seed(42)
import time, openai, json
from ..grm.prediction_function import generate

PRINCIPLES = {
    "rewardbench": "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response",
    "rewardbench-2": "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response"
}

JUDGE_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors {specific_principle}. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better. \n\n"
    "[Instruction]\n{instruction}\n[The Start of Assistant A's Answer]\n{output_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{output_2}\n[The End of Assistant B's Answer]"
)

def create_judgment_prompts(reward_data, judge_config):
    judge_principle = PRINCIPLES[judge_config["criteria"]]
    prompts = []
    caseid = 0
    for case in reward_data:
        user_prompt = JUDGE_PROMPT.format(
            instruction=case["instruction"],
            output_1=case["response_a"],
            output_2=case["response_b"],
            specific_principle=judge_principle,
        )
        prompts.append(
            {
                "id": str(caseid),
                "messages":[{"role": "user", "content": user_prompt}],
            }
        )
        caseid+=1
    return prompts

def naive(reward_data, llm_model_config, judge_config, reward_benchmark):
    # generate judgment
    judgment_prompts = create_judgment_prompts(reward_data, judge_config)
    judgment_generation = generate(judgment_prompts, llm_model_config)
    for source, target in zip(reward_data, judgment_generation):
        source["naive_judgment"] = target
    return reward_data, "naive_judgment"
