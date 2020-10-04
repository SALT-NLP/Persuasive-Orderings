import nltk
import random

def fix_bad(bad, good, target_sents, delete=False, insert=False):
    bad_copy = dict(bad)
    avg_sents = []
    
    count = 0
    for k in bad_copy:
        _, good_item = random.choice(list(good.items()))
        sents = nltk.sent_tokenize(good_item)
        sents_bad = nltk.sent_tokenize(bad_copy[k])
        avg_sents.append(len(sents_bad))
        
        insertions = sents[-2:]
        target = target_sents[k]
        r = sents[-2:]
        
        temp = (" ".join(sents_bad[:target - 1])) + (" ".join(sents_bad[target+2:])) if delete else bad_copy[k]
        if insert: temp += ". " + " ".join(insertions)   
        bad_copy[k] = temp

    return bad_copy

def get_samples(good_strats, bad_strats, merged_strats, 
            unlabeled_data_pkl, labeled_data_pkl, all_strategies, inverse_label_mapping):
    all_good = {}
    all_bad = {}
    for k in merged_strats:
        samp_keys = merged_strats[k][-1]
        converted_samp = ""
        for curr_strat in k.split(" "):
            if curr_strat == "UNK":
                converted_samp += curr_strat + " "
            else:
                converted_samp += all_strategies[inverse_label_mapping[int(curr_strat)]] + " "
                
        if k in good_strats:
            for key in samp_keys:
                if key in labeled_data_pkl:
                    all_good[key] = labeled_data_pkl[key]
                elif key in unlabeled_data_pkl:
                    all_good[key] = unlabeled_data_pkl[key]
        elif k in bad_strats:
            for key in samp_keys:
                if key in labeled_data_pkl:
                    all_bad[key] = labeled_data_pkl[key]
                elif key in unlabeled_data_pkl:
                    all_bad[key] = unlabeled_data_pkl[key]
                    
    for k in all_good:
        if type(all_good[k]) is list:
            all_good[k] = " ".join(all_good[k][0])
    for k in all_bad:
        if type(all_bad[k]) is list:
            all_bad[k] = " ".join(all_bad[k][0])
    return all_bad, all_good