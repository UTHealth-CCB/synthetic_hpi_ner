"""Script to build words, chars and tags vocab"""

__author__ = "Guillaume Genthial"

from collections import Counter
from pathlib import Path

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1
import sys

if __name__ == '__main__':
    print('argv:{}'.format(sys.argv))
    if len(sys.argv) == 2:
        data_dir = sys.argv[1]
    else:
        data_dir = './data/i2b2_n2c2_synth/fold_1/input'

    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return '{}/{}.words.txt'.format(data_dir, name)

    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'testa', 'testb']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path('{}/vocab.words.txt'.format(data_dir)).open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path('{}/vocab.chars.txt'.format(data_dir)).open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Tags
    # Get all tags from the training set, should contain at least dev set  also , or, may cause error due to limited train set not containing tages which appeared in dev
    # e.g., in SemEval transfer learning, to test limited data set.

    def tags(name):
        return '{}/{}.tags.txt'.format(data_dir, name)

    print('Build vocab tags (may take a while)')
    vocab_tags = set()

    # include all train, dev, test sets for all the possible tags!
    for n in ['train', 'testa', 'testb']:
        with Path(tags(n)).open() as f:
            for line in f:
                vocab_tags.update(line.strip().split())

    #with Path(tags('train')).open() as f:
    #    for line in f:
    #        vocab_tags.update(line.strip().split())

    with Path('{}/vocab.tags.txt'.format(data_dir)).open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))
