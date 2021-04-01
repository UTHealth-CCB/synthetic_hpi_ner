#!/usr/bin/python
import sys
import os
import argparse

# calculate_scores: calculate gold and predict scores
# exact: False by default, calculate both exact and inexact match results
#        else, calculate only exact results
def calculate_scores( gold_span, predict_span, exact=True):
    right = 0
    right_gold = 0
    right_predict = 0

    for s1, e1 in gold_span:
        for s2, e2 in predict_span:
            if s1 == s2 and e1 == e2:
                right += 1
                break

    for s1, e1 in gold_span:
        for s2, e2 in predict_span:
            if ( s2 <= s1 and s1 < e2 ) or ( s2 < e1 and e1 <= e2 ) or ( s1 <= s2 and s2 < e1 ) or ( s1 < e2 and e2 <= e1 ):
                right_gold += 1
                break

    for s1, e1 in predict_span:
        for s2, e2 in gold_span:
            if ( s2 <= s1 and s1 < e2 ) or ( s2 < e1 and e1 <= e2 ) or ( s1 <= s2 and s2 < e1 ) or ( s1 < e2 and e2 <= e1 ):
                right_predict += 1
                break
    if predict_span:
        p = float(right) / len( predict_span )
    else:
        p = 0.0
    if gold_span:
        r = float(right) / len( gold_span )
    else:
        r = 0.0
    if p == 0.0 or r == 0.0:
        f = 0.0
    else:
        f = 2 * p * r / ( p + r )
    if predict_span:
        p2 = float(right_gold) / len( predict_span )
    else:
        p2 = 0.0
    if gold_span:
        r2 = float(right_predict) / len( gold_span )
    else:
        r2 = 0.0
    if p2 == 0.0 or r2 == 0.0:
        f2 = 0.0
    else:
        f2 = 2 * p2 * r2 / ( p2 + r2 )

    if not exact:
        return '%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d' % (p, r, f, p2, r2, f2, right, right_predict, right_gold, len( predict_span ), len( gold_span ) )
    else:
        return '%.3f\t%.3f\t%.3f\t%d\t%d\t%d' % (p, r, f, right, len( predict_span ), len( gold_span ) )

def calculate_scores_micro_overall( gold, predict,exact=True):
    right = 0
    right_gold = 0
    right_predict = 0

    for k in gold:
        gold_span=gold[k]
        if k not in predict:
            predict_span=[]
        else:
            predict_span=predict[k]
        for s1, e1 in gold_span:
            for s2, e2 in predict_span:
                if s1 == s2 and e1 == e2:
                    right += 1
                    break
    for k in gold:
        gold_span=gold[k]
        if k not in predict:
            predict_span=[]
        else:
            predict_span=predict[k]
        for s1, e1 in gold_span:
            for s2, e2 in predict_span:
                if ( s2 <= s1 and s1 < e2 ) or ( s2 < e1 and e1 <= e2 ) or ( s1 <= s2 and s2 < e1 ) or ( s1 < e2 and e2 <= e1 ):
                    right_gold += 1
                    break

    for k in gold:
        gold_span=gold[k]
        if k not in predict:
            predict_span=[]
        else:
            predict_span=predict[k]
        for s1, e1 in predict_span:
            for s2, e2 in gold_span:
                if ( s2 <= s1 and s1 < e2 ) or ( s2 < e1 and e1 <= e2 ) or ( s1 <= s2 and s2 < e1 ) or ( s1 < e2 and e2 <= e1 ):
                    right_predict += 1
                    break

    all_predict= sum([len(v) for k, v in predict.items()])
    all_gold = sum([len(v) for k, v in gold.items()])
    if all_predict > 0:
        p = float(right) / all_predict
    else:
        p = 0.0
    if all_gold > 0:
        r = float(right) / all_gold
    else:
        r = 0.0
    if p == 0.0 or r == 0.0:
        f = 0.0
    else:
        f = 2 * p * r / ( p + r )

    if all_predict > 0:
        p2 = float(right_gold) / all_predict
    else:
        p2 = 0.0
    if all_gold > 0:
        r2 = float(right_predict) / all_gold
    else:
        r2 = 0.0
    if p2 == 0.0 or r2 == 0.0:
        f2 = 0.0
    else:
        f2 = 2 * p2 * r2 / ( p2 + r2 )

    if not exact:
        return '%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%d\t%d\t%d' % (p, r, f, p2, r2, f2, right, right_predict, right_gold, all_predict, all_gold)
    else:
        return '%.3f\t%.3f\t%.3f\t%d\t%d\t%d' % (p, r, f, right, all_predict, all_gold )

# load spans: support eval_type as 'ner' or 'classification'
def load_spans( labels, eval_type='ner' ):
    spans = {}
    for i in range( len( labels ) ):
        if eval_type == 'classification':
            label = labels[i]
            spans.setdefault( label, [] )
            spans[label].append((i, i+1))
        else:
            label = labels[i]
            if type(label) == list:
                print('Warning: label is list {}'.format(label))
            if label.startswith( 'B-' ):
                s = i
                e = i + 1
                sem = label[ 2: ]
                # found an entity
                for j in range( i + 1, len( labels ) ):
                    e = j
                    if labels[j] != 'I-' + sem:
                        break
                spans.setdefault( sem, [] )
                spans[ sem ].append( ( s, e ) )
            if label.startswith( 'DB-' ):
                s = i
                e = i + 1
                sem = label[ 3: ]
                # found an entity
                for j in range( i + 1, len( labels ) ):
                    e = j
                    if labels[j] != 'DI-' + sem:
                        break
                spans.setdefault( sem, [] )
                spans[ sem ].append( ( s, e ) )

            if label.startswith( 'HB-' ):
                s = i
                e = i + 1
                sem = label[ 3: ]
                # found an entity
                for j in range( i + 1, len( labels ) ):
                    e = j
                    if labels[j] != 'HI-' + sem:
                        break
                spans.setdefault( sem, [] )
                spans[ sem ].append( ( s, e ) )

    return spans

# result_file format: X ... GOLD_LABEL PRED_LABEL
def load_combined_bio( result_file, sep_tag='\t' ):
    gold = []
    predict = []
    line_comments = ''
    with open( result_file,'r',encoding='utf-8' ) as infile:
        for line in infile:
            if line.strip() == '':
                gold.append( 'O' )
                predict.append( 'O' )
                continue
            if (line.strip().startswith('###')) and (line.strip().endswith('$$$')):
                line_comments = line.strip()
                continue
            cols = line.strip().split(sep_tag)
            if len(cols) < 3:
                print(line_comments.strip('\n'))
                print('Warning: too few columns in lines {}\n'.format(line.strip('\n')))
            else:
                p = cols[-1] 
                predict.append( p.split()[0] )
                g = cols[-2]                
                gold.append(g)

    return gold, predict

def load_bio( result_file, sep_tag='\t' ):
    result = []
    with open( result_file ) as infile:
        for line in infile:
            if line.strip() == '':
                result.append( 'O' )
                continue
            # ignore disjoint entities for now
            #line = line.replace( 'B-DDisease_Disorder', 'B-Disease_Disorder' )
            #line = line.replace( 'B-HDisease_Disorder', 'B-Disease_Disorder' )
            #line = line.replace( 'I-DDisease_Disorder', 'I-Disease_Disorder' )
            #line = line.replace( 'I-HDisease_Disorder', 'I-Disease_Disorder' )
            result.append( line.split(sep_tag)[-1] )

    return result

# evaluate gold and pre label file
# label_file: gold_pred combined label file if type(label_file) is str, 
#             else will be taken as list/sequence to contains gold and pred label files sequently
# eval_score_file: write eval_score_file if not empty
# exact: False by default, calculate both exact and inexact match results
#        else, calculate only exact results
# sep_tag: seperated tag between label file's items in each line, '\t' by default
# labels: the pre-ordered-specified labels to evaluate, empty to evaluate re-ordered labels by default
def evaluate(label_file, eval_score_file='', sep_tag_type='tab', eval_type='ner', exact=False, labels=None):    
    print(f'evaluate: {label_file} {eval_score_file} {sep_tag_type} {eval_type} {exact} {labels}')
    sep_tag = ' ' if sep_tag_type=='space' else '\t'    
    try:
        if type(label_file) is str:
            gold, predict = load_combined_bio( label_file, sep_tag)
        elif len(label_file) == 2:
            gold_label_file = label_file[0]
            pred_label_file = label_file[1]
            gold = load_bio(gold_label_file, sep_tag)
            predict = load_bio(pred_label_file, sep_tag)
        else:
            print('Waring: error label_file format, ignoring evaluation...')
            return
    except Exception as e:
        print(e)
        return
    gold_spans = load_spans( gold, eval_type )
    predict_spans = load_spans( predict, eval_type )

    if not labels:
        labels = list({item[2:] for item in set(gold).union(set(predict)) if item[2:] and item != 'O'})
        labels.sort()
    else:
        labels = [item.strip() for item in labels.split(',') if item.strip()]
    print('\n\n\n')
    eval_results = []
    if exact:
        score_cols = 'P\tR\tF1\tright\tpredict\tgold\tSemantic'
        print(score_cols)
        eval_results.append(score_cols)
    else:
        score_cols = 'P(exact)\tR(exact)\tF1(exact)\tP(relax)\tR(relax)\tF1(relax)\tright\tright_predict\tright_gold\tpredict\tgold\tSemantic'
        print(score_cols)
        eval_results.append(score_cols)

    for k in sorted(gold_spans.keys(),key=lambda x:labels.index(x)):
        gold_span = gold_spans[k]
        if k in predict_spans:
            predict_span = predict_spans[k]
        else:
            predict_span = []
        scores =  calculate_scores( gold_span, predict_span,exact=exact ) + '\t' + k
        print (scores)
        eval_results.append(scores)
    
    #gold_spans={k:v for k,v in gold_spans.items() if k not in {'Anatomic_location','Negation_cue'}}
    gold_spans={k:v for k,v in gold_spans.items()}
    #predict_spans={k:v for k,v in predict_spans.items() if k not in {'Anatomic_location','Negation_cue'}}
    predict_spans={k:v for k,v in predict_spans.items()}
    scores = calculate_scores_micro_overall(gold_spans,predict_spans,exact=exact) + '\t' + 'overall' 
    print(scores)
    eval_results.append(scores)
    if eval_score_file:
        with open(eval_score_file, 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(eval_results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', '--labels_file', type=str, default='', help='combined gold and pred label file')
    parser.add_argument('-gl', '--gold_label_file', type=str, default='', help='gold label file, used together with -pl, ignored if -fl defined and not empty')
    parser.add_argument('-pl', '--pred_label_file', type=str, default='', help='prediction label file, used together with -gl, ignored if -fl defined and not empty')
    parser.add_argument('-ef', '--eval_score_file', type=str, default='', help='evaluate score file')    
    parser.add_argument('-st', '--sep_tag_type', type=str, default='\t', help='seperated tag type between items in pred/gold file, i.e., tab or space')
    parser.add_argument('-et', '--eval_type', type=str, default='ner', help='evaluation type, i.e., ner or classification')
    parser.add_argument('-e', '--exact', type=bool, default=False, help='only calculate exact result, or calculate both exact and inexact results')
    parser.add_argument('-l', '--labels', type=str, default='', help='evaluated labels seperated with ",", e.g., test, -l problem,treatment')
    
    args = parser.parse_args()
    if args.labels_file:
        evaluate(args.labels_file, args.eval_score_file, args.sep_tag_type, args.eval_type, args.exact, args.labels)
    elif args.gold_label_file and args.pred_label_file:
        labels_file = (args.gold_label_file, args.pred_label_file)
        evaluate(labels_file, args.args.sep_tag_type, args.eval_score_file, args.sep_tag, args.eval_type, args.exact, args.labels)
    else:
        print('Invalid parameters:\n\npython evaluation.py -lf gold_pred_label.txt\n')
        print('Or:\npython evaluation.py -gl gold_label.txt -pl pred_label.txt')
        # test evaluation
        print('run test...')
        test_evaluation()
    print('evaluate done.')