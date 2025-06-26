def evaluate_reward(reward_data, output_key):
    right = 0
    count=0
    length = 0
    item = 2
    for case in reward_data:
        try:
            if ("[[A]]" in case[output_key] and case["label"] == 0) or ("[[B]]" in case[output_key] and case["label"] == 1):
                right+=1
        except:
            continue
        count+=1
    print("ACC: ", right/count)
    print("COUONT: ", count)