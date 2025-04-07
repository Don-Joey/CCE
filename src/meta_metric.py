def compute_acc(judgments):
    correct = 0
    for judgment in judgments:
        if '[[A]]' in judgment["judgment_output"] and judgment["label"] == 1:
            correct+=1
        elif '[[B]]' in judgment["judgment_output"] and judgment["label"] == 2:
            correct+=1
    print("Final Acc:", correct/len(judgments))

def compute_corr(judgments):
    pass