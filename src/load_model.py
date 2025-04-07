from .openai_api import Judge_Api_configuration, Generation1_Api_configuration, Generation2_Api_configuration
from .local_model import Generation1_Local_configuration, Judge_Local_configuration, Generation2_Local_configuration
IS_API = ["gpt-4o", "gpt-4o-mini", "o1-preview"]

def load_generation_model1(config):
    if config["generation_llm1"] == "default":
        return None
    if config["generation_llm1"]["model_name"] in IS_API:
        model = Generation1_Api_configuration("E:\self_consistency_judge\datafolder", config)
    else:
        model = Generation1_Local_configuration("E:\self_consistency_judge\datafolder", config)
    return model

def load_generation_model2(config):
    if config["generation_llm2"] == "default":
        return None
    if config["generation_llm2"]["model_name"] in IS_API:
        model = Generation2_Api_configuration("E:\self_consistency_judge\datafolder", config)
    else:
        model = Generation2_Local_configuration("E:\self_consistency_judge\datafolder", config)
    return model

def load_judge_model(config):
    if config["judge_llm"]["model_name"] in IS_API:
        model = Judge_Api_configuration("E:\self_consistency_judge\datafolder", config)
    else:
        model = Judge_Local_configuration("E:\self_consistency_judge\datafolder", config)
    return model
