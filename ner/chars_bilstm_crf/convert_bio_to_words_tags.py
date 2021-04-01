# -*- coding: utf-8 -*-

import os
import sys
import re

NON_ASCII_RE = re.compile(r'[^\x00-\x7F]+')
TAG_SEP = '\t'

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) == 4:        
        DATADIR = sys.argv[1]
        EXT = sys.argv[2] 
        # filtered label sparately with ','
        LABEL_FILTERED = sys.argv[3].split(',')   
    elif len(sys.argv) == 3:
        DATADIR = sys.argv[1]
        EXT = sys.argv[2]
        LABEL_FILTERED = []
    elif len(sys.argv) == 2:
        DATADIR = sys.argv[1]#'../data/i2b2_n2c2_synth/fold_1/input'
        EXT = '.txt'
        LABEL_FILTERED = []
    else:
        DATADIR = './data/i2b2_n2c2_synth/fold_1/input'
        EXT = '.bio'
        LABEL_FILTERED = ['drug']
    print('DATADIR:{}'.format(DATADIR))    
    
    in_out_names = {"train":"train", "dev":"testa", "test":"testb"}
    for [in_name,out_name] in in_out_names.items():
        in_pfn = os.path.join(DATADIR, '{}{}'.format(in_name, EXT))
        out_words_fpn = os.path.join(DATADIR, '{}.words.txt'.format(out_name))
        out_tags_fpn = os.path.join(DATADIR, '{}.tags.txt'.format(out_name))
        words = ''
        tags = ''
        all_words = []
        all_tags = []        
        with open(in_pfn, 'r') as in_file:            
            for line in in_file:
                #print(line)
                sl = line.strip()
                if sl == '':
                    # new line for end of new senence
                    words = ''
                    tags = ''
                    continue
                word = sl.split(TAG_SEP)[0]
                tag = sl.split(TAG_SEP)[-1]
                if len(word.split(' '))>1:
                    raise RuntimeError('Error for xmi2bio tokenization: {}'.format(word))
                if words == '' and tags == '':
                    # begin of new sentence                    
                    #if words.find('emboli') >= 0:
                    #    print(words)
                    #NON_ASCII_RE.sub('_', words)                    
                    words = word
                    if tag.split('-')[-1] not in LABEL_FILTERED:                        
                        tags = tag
                    else:
                        tags = 'O'                
                    # append the first word&tag to all_words&all_tags
                    all_words.append(words)
                    all_tags.append(tags)
                else:
                    # in the sentence, update words and tags
                    words = words + ' ' + word
                    if tag.split('-')[-1] not in LABEL_FILTERED:                        
                        tags = tags + ' ' + tag
                    else:
                        # set LABEL_FILTERED to 'O'
                        tag = 'O'
                        tags = tags + ' ' + tag
                    all_words[len(all_words)-1] = words
                    all_tags[len(all_tags)-1] = tags
        with open(out_words_fpn, 'w') as out_file:
            for words in all_words:
                out_file.write('{}\n'.format(words))
        with open(out_tags_fpn, 'w') as out_file:
            for tags in all_tags:
                out_file.write('{}\n'.format(tags))
        
    print('Convert bio files done!')