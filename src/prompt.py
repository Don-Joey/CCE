import json
import os
import copy
import time as time
import random

import anthropic
import google.generativeai as genai
import openai
from fastchat.conversation import get_conv_template
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from openai import OpenAI
from together import Together

SEQUENCIAL = ["gpt-4o", "gpt-4o-mini", "llama-3.1-8b-instruct", "qwen-2.5-7b-instruct", "llama-3.2-3b-instruct", "qwen-2.5-3b-instruct", "llama-3.2-1b-instruct", "qwen-2.5-1.5b-instruct", "qwen-2.5-0.5b-instruct"]

ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
)

# feel free to add more models to this list via PR
# available models: https://docs.together.ai/docs/inference-models
TOGETHER_MODEL_LIST = (
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
    "google/gemma-2-27b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
)

GEMINI_MODEL_LIST = (
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-exp-0801",
    "gemini-1.5-pro-exp-0827",
    "gemini-1.5-flash-exp-0827",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-exp-0827",
)

API_MODEL_LIST = OPENAI_MODEL_LIST + ANTHROPIC_MODEL_LIST + TOGETHER_MODEL_LIST + GEMINI_MODEL_LIST


# API setting constants
API_MAX_RETRY = 25
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

#prompt_v2 = (
#    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
#    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
#    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
#    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
#    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
#    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
#    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # removed tie option as , and \"[[C]]\ " for a tie
#)

prompt_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors "
    "such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # removed tie option as , and \"[[C]]\ " for a tie
)
lfqa_prompt_v2 = (
    "You are a helpful assistant in evaluating the quality of the responses for a given instruction. The responses being evaluated are likely longer form responses to questions requiring in-depth reasoning.\n"
    "Your goal is to select the best response. Select Response A or Response B, that is better for the given instruction. Do NOT say both / neither are good.\n "
    "Here are some rules of the evaluation: "
    "(1) Consider how each response satisfies the instruction SEPARATELY. Because the instructions are often open-ended and complex questions, answers may differ between responses. This means that the content in response A should not be used to say that the content in the response B is wrong, and vice versa. "
    "(2) You should consider the responses carefully, paying attention to the thoroughness and completeness of the reasoning and factuality. The response should correct any false assumptions in the question when present and address the complexity of questions with no set answer. "
    "(3) The response should consider all aspects of the question and be well formulated and easy to follow. "
    "(4) The response should not contain irrelevant information or factually incorrect information or common misconceptions "
    "(5) Ensure that you respond with the response you think is better after giving your reasoning.\n"
    "Your reply should strictly follow this format: **Reasoning:** <feedback evaluating the responses> \n"
    "**Result:** <[[A]] or [[B]]>\n" 
    "Here is the data.\n"
)

judgebench_prompt_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants, A and B, to the user question displayed below. "
    "The two responses "
    "are generated by two different AI assistants A and B respectively. Your task is to decide which assistant answers the question better.  "
    "Your evaluation should consider helpfulness, relevance and conciseness. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: '[[A]]' if assistant A is better, '[[B]]' if assistant B is better.\n "
)


prompt_helpsteer_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors "
    "helpfulness, correctness, coherence, complexity, verbosity. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # removed tie option as , and \"[[C]]\ " for a tie
)

# used for gemini pro llm as a judge (API implementation coming soon)
# implementation details shared from Gemini Alignment Team
# usage is as follows:
# -> no system prompt
# -> use following text, followed by instruction then example. E.g.
# [Rating instructions]
# [Prompt]: [Instruction1]
prompt_v2_gemini = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. "
    "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "
    "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "
    "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. "
    "Be as objective as possible. "
    "Your output should only consist of '[[A]]' if assistant A is better, or '[[B]]' if assistant B is better. Omit any other output.\n"
)

prompt_multi_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. "
    "You should focus on who provides a better answer to the second user question. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'  # removed tie option as , and \"[[C]]\" for a tie
)

MTBENCH_V2 = {
    "name": "pair-v2",
    "type": "pairwise",
    "system_prompt": prompt_v2,
    "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]",
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

LFQA_V2 = {
    "name": "pair-v2",
    "type": "pairwise",
    "system_prompt": lfqa_prompt_v2,
    "prompt_template": "[Instruction]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]",
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

MTBENCH_HELPSTEER_V2 = {
    "name": "pair-v2",
    "type": "pairwise",
    "system_prompt": prompt_helpsteer_v2,
    "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]",
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

MTBENCH_MULTI_V2 = {
    "name": "pair-v2-multi-turn",
    "type": "pairwise",
    "system_prompt": prompt_multi_v2,
    "prompt_template": (
        "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n"
        "### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n"
        "<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n"
        "### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>"
    ),
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

JUDGEBENCH_V2 = {
    "name": "pair-v2",
    "type": "pairwise",
    "system_prompt": judgebench_prompt_v2,
    "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer] # Which is better, [[A]] or [[B]]? Your response should be either \"[[A]]\" or \"[[B]]\":",
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

# Prometheus prompts taken from
# https://github.com/prometheus-eval/prometheus-eval/blob/becd223d624896a5383e5dd9b766d740261a80f2/eval/prompts.py
RELATIVE_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{orig_instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Score Rubric:
{score_rubric}

###Feedback: """

# Absolute grading prompt w/o refernece answer for a2r
ABSOLUTE_PROMPT = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Score Rubrics:
{score_rubric}

###Feedback: """

AUTOJ_COARSE_SCORE_RUBRIC = """
[Is the given response reflective of helpfulness, harmlessness, and honesty?]
Score 1: The response is misleading, harmful, or dishonest, negatively impacting the user.
Score 2: The response tries to be helpful and honest but might mislead or be somewhat harmful, requiring further clarification or caution.
Score 3: The response is honest and aims to be helpful, but could improve in avoiding harm and fully ensuring accuracy.
Score 4: The response is helpful, honest, and largely avoids harm, providing accurate information with some ethical considerations.
Score 5: The response is outstanding in its helpfulness, honesty, and harmlessness, offering thorough, accurate advice with proactive ethical safeguards."""

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

OFFSETBIAS_PROMPT = """You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.

Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.
Do NOT provide any explanation for your choice.
Do NOT say both / neither are good.
You should answer using ONLY "Output (a)" or "Output (b)". Do NOT output any other words.
Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the outputs were presented should NOT affect your judgment, as Output (a) and Output (b) are **equally likely** to be the better.

# Instruction:
{instruction}
# Output (a):
{output_1}
# Output (b):
{output_2}
# Which is better, Output (a) or Output (b)? Your response should be either "Output (a)" or "Output (b)":"""

CON_J_PROMPT = """作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在连贯性、准确性、覆盖度和上述定义的整体质量方面最为符合。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{{"原因": "你的解释", "更好的回答": 1}}。以下是问题和候选回答的内容：
    \n问题：{instruction}
回答1：{output_1}
回答2：{output_2}"""

DEFAULT_GENERATE = {
    "name": "default_generate",
    "type": "generate",
    "system_prompt": None, 
    "prompt_template": "{question}\n",
    "description": "Prompt for general questions",
    "category": "general",
}


def openai_template(system_prompt, user_prompt):
    template = "chatgpt"
    conv = get_conv_template(template)

    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    if system_prompt:
        conv.set_system_message(system_prompt)
    messages = conv.to_openai_api_messages()
    return messages

# format with prompt_template.format(question=question, answer_a=answer_a, answer_b=answer_b)
def format_judge_answers(instruction, answer_a, answer_b, multi_turn=False, model_modifier=None):
    kwargs = {}
    if model_modifier == "prometheus":
        if multi_turn:
            raise ValueError("Prometheus prompts do not support multi-turn prompts")
        else:
            system_prompt = REL_SYSTEM_PROMPT
            user_prompt = RELATIVE_PROMPT.format(
                orig_instruction=instruction,
                response_A=answer_a[1]["content"],
                response_B=answer_b[1]["content"],
                score_rubric=AUTOJ_COARSE_SCORE_RUBRIC,
                **kwargs,
            )
    elif model_modifier == "Con-J":
        if multi_turn:
            raise ValueError("Con-J prompts do not support multi-turn prompts")
        else:
            system_prompt = ""
            user_prompt = CON_J_PROMPT.format(
                instruction=instruction, output_1=answer_a[1]["content"], output_2=answer_b[1]["content"]
            )
    elif model_modifier == "offsetbias":
        if multi_turn:
            raise ValueError("Offsetbias prompts do not support multi-turn prompts")
        else:
            system_prompt = ""
            user_prompt = OFFSETBIAS_PROMPT.format(
                instruction=instruction, output_1=answer_a[1]["content"], output_2=answer_b[1]["content"]
            )
    elif model_modifier == "helpsteer2":
        if multi_turn:
            system_prompt = MTBENCH_MULTI_V2["system_prompt"]
            user_prompt = MTBENCH_MULTI_V2["prompt_template"].format(
                question_1=instruction,
                question_2=answer_a[2]["content"],
                answer_a_1=answer_a[1]["content"],
                answer_b_1=answer_b[1]["content"],
                answer_a_2=answer_a[3]["content"],
                answer_b_2=answer_b[3]["content"],
                **kwargs,
            )
        else:
            system_prompt = MTBENCH_HELPSTEER_V2["system_prompt"]
            user_prompt = MTBENCH_HELPSTEER_V2["prompt_template"].format(
                question=instruction,
                answer_a=answer_a[1]["content"],
                answer_b=answer_b[1]["content"],
                **kwargs,
            )
    elif model_modifier == "lfqa":
        system_prompt = LFQA_V2["system_prompt"]
        user_prompt = LFQA_V2["prompt_template"].format(
            question=instruction,
            answer_a=answer_a[1]["content"],
            answer_b=answer_b[1]["content"],
            **kwargs,
        )
    elif model_modifier == "judgebench":
        system_prompt = JUDGEBENCH_V2["system_prompt"]
        user_prompt = JUDGEBENCH_V2["prompt_template"].format(
            question=instruction,
            answer_a=answer_a[1]["content"],
            answer_b=answer_b[1]["content"],
            **kwargs,
        )
    else:
        if multi_turn:
            system_prompt = MTBENCH_MULTI_V2["system_prompt"]
            user_prompt = MTBENCH_MULTI_V2["prompt_template"].format(
                question_1=instruction,
                question_2=answer_a[2]["content"],
                answer_a_1=answer_a[1]["content"],
                answer_b_1=answer_b[1]["content"],
                answer_a_2=answer_a[3]["content"],
                answer_b_2=answer_b[3]["content"],
                **kwargs,
            )
        else:
            system_prompt = MTBENCH_V2["system_prompt"]
            user_prompt = MTBENCH_V2["prompt_template"].format(
                question=instruction,
                answer_a=answer_a[1]["content"],
                answer_b=answer_b[1]["content"],
                **kwargs,
            )

    # gemini adds what was the system prompt before the content, and has no system prompt
    if model_modifier == "gemini":
        user_prompt = prompt_v2_gemini + user_prompt
        system_prompt = None
    return system_prompt, user_prompt

def generate_judge_prompts(config, pairs):
    prompts = []
    for pair in pairs:
        if config["w_label"] == False and (config["generation_llm1"]!="default" and config["generation_llm2"]!="default"):
            print(">>>>>>>>>>>>>>>>>>>>")
            exit()
            if random.uniform(0, 1) < 0.5:
                temp = copy.copy(pair["answer_a"])
                pair["answer_a"] = pair["answer_b"]
                pair["answer_b"] = temp
                pair["label"] = 2 if SEQUENCIAL.index(config["generation_llm1"]["model_name"]) < SEQUENCIAL.index(config["generation_llm2"]["model_name"]) else 1
            else:
                pair["label"] = 1 if SEQUENCIAL.index(config["generation_llm1"]["model_name"]) < SEQUENCIAL.index(config["generation_llm2"]["model_name"]) else 2
        elif config["w_label"] == False:
            del pair["label"]
        
        system_prompt, user_prompt = format_judge_answers(pair["instruction"], pair["answer_a"], pair["answer_b"], multi_turn=pair["multiturn"], model_modifier=config["judgment_prompt"])
        prompt = (system_prompt, user_prompt)
        
        prompt = openai_template(system_prompt, user_prompt)
        prompts.append(prompt)
        pair["judge_prompt"] = prompt
    
    return prompts

# format with prompt_template.format(question=question, answer_a=answer_a, answer_b=answer_b)
def format_response_answers(instruction, multi_turn=False, model_modifier=None):
    kwargs = {}
    if multi_turn:
        system_prompt = DEFAULT_GENERATE["system_prompt"]
        user_prompt = DEFAULT_GENERATE["prompt_template"].format(
            question=instruction,
            **kwargs,
        )
    else:
        system_prompt = DEFAULT_GENERATE["system_prompt"]
        user_prompt = DEFAULT_GENERATE["prompt_template"].format(
            question=instruction,
            **kwargs,
        )
    return system_prompt, user_prompt


def generate_response_prompts(config, pairs):
    prompts = []
    for pair in pairs:
        system_prompt, user_prompt = format_response_answers(pair["instruction"], multi_turn=pair["multiturn"], model_modifier="gpt-4o")
        prompt = (system_prompt, user_prompt)
        prompt = openai_template(system_prompt, user_prompt)
        prompts.append(prompt)
        pair["judge_prompt"] = prompt
    
    return prompts