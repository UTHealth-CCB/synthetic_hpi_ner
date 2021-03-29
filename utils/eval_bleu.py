import os
import nltk
import random


def bleu_eval_func(seq_len, reference_file, hypothesis_file, max_gram=2, para_len=1):
    #################################################
    reference = []
    para_count = 0
    with open(reference_file, encoding='utf-8') as fin:
        for line in fin:
            line = line.split()
            line = line[0:seq_len]
            if (not line):
                continue
            while line[-1] == str(0):
                line.remove(str(0))
            reference.append(line)
            para_count += 1
            if (para_count>para_len):
                break
    #################################################
    hypothesis = []
    with open(hypothesis_file, encoding='utf-8') as fin:
        for line in fin:        
            line = line.split()
            line = line[0:seq_len]          
            if (not line):
                continue            
            #print(line)
            while line[-1] == str(0):
                line.remove(str(0))
                if not line:
                    break
            hypothesis.append(line)

    random.shuffle(hypothesis)
    #################################################

    bleu_score = []
    for ngram in range(2, max_gram + 1):
        weight = tuple((1. / ngram for _ in range(ngram)))
        bleu_list = []
        for h in hypothesis[:2000]:
            smoothing_func = nltk.translate.bleu_score.SmoothingFunction()
            BLEUscore = nltk.translate.bleu_score.sentence_bleu(reference, h, weight,
                                                                smoothing_function=smoothing_func.method1)
            bleu_list.append(BLEUscore)
        print(len(weight), '-gram BLEU score: ', 1.0 * sum(bleu_list) / len(bleu_list))
        bleu_score.append(1.0 * sum(bleu_list) / len(bleu_list))

    return bleu_score