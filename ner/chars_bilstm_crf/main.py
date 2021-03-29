"""GloVe Embeddings + chars bi-LSTM + bi-LSTM + CRF

__author__ = "Guillaume Genthial"

01272020, jli34@UTH, update for transfer learning on ner tasks for I2B2-2010&SemEval-2015
"""

import functools
import json
import logging
from pathlib import Path
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

#DATADIR = '../../data/example'
# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for NER task')

parser.add_argument('--gpu', type=str, default='0', help='available gpu number')

default_data_dir = 'data/ner_i2b2/source1/' #data/ner_i2b2/target1_transfer_learning/fold_1/input
parser.add_argument('--data_dir', type=str, default=default_data_dir +'/input', help='train raw_data source')
parser.add_argument('--model_dir', type=str, default=default_data_dir + '/model_ep150', help='model dir to save/load checkpoint')
parser.add_argument('--output_dir', type=str, default=default_data_dir + '/output_ep150', help='output dir')
parser.add_argument('--epoch', type=int, default=30, help='#epoch of training')
#parser.add_argument('--warm_start_dir', type=str, default='data/ner_i2b2/source1/model', help='#warm start for initial checkpoint')
parser.add_argument('--warm_start_dir', type=str, default=None, help='#warm start for initial checkpoint')


#parser.add_argument("--transformer",type=str,default="False",help="transer learning")
#parser.add_argument('--pretrain_embedding', type=str, default='random',help='use pretrained char embedding or init it randomly')

#parser.add_argument('--batch_size', type=int, default=32, help='#sample of each minibatch')

#parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
#parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
#parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
#parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
#parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
#parser.add_argument('--dropout', type=float, default=0.6, help='dropout keep_prob')
#parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
#parser.add_argument('--pretrain_embedding', type=str, default='emr-bert-embedding_mat.npy',help='use pretrained char embedding or init it randomly')
#parser.add_argument('--embedding_dim', type=int, default=768, help='random init char embedding_dim')
#parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training raw_data before each epoch')
#parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
#parser.add_argument('--demo_model', type=str, default='1565801182', help='model for test and demo')#random_char_300,1524919794
#parser.add_argument('--model', type=str, default='ner_i2b2_target1', help='model for train, test and demo')#random_char_300,1524919794
#parser.add_argument('--embedding_dir', type=str, default='word2vector', help='embedding files dir')
#parser.add_argument("--index",type=str,default='trainData/index.txt')
#parser.add_argument("--index",type=str,default='ner_i2b2_index.txt')

# Logging
#Path('results').mkdir(exist_ok=True)
#tf.logging.set_verbosity(logging.INFO)
#handlers = [
#    logging.FileHandler('results/main.log'),
#    logging.StreamHandler(sys.stdout)
#]
#logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),               # (words, nwords)
               ([None, None], [None])),    # (chars, nchars)
              [None])                      # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    (words, nwords), (chars, nchars) = features
    dropout = params['dropout']
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embeddings', [num_chars, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
                                        training=training)

    # Char LSTM
    dim_words = tf.shape(char_embeddings)[1]
    dim_chars = tf.shape(char_embeddings)[2]
    flat = tf.reshape(char_embeddings, [-1, dim_chars, params['dim_chars']])
    t = tf.transpose(flat, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['char_lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    _, (_, output_fw) = lstm_cell_fw(t, dtype=tf.float32,
                                     sequence_length=tf.reshape(nchars, [-1]))
    _, (_, output_bw) = lstm_cell_bw(t, dtype=tf.float32,
                                     sequence_length=tf.reshape(nchars, [-1]))
    output = tf.concat([output_fw, output_bw], axis=-1)
    char_embeddings = tf.reshape(output, [-1, dim_words, 50])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)
 
def write_predictions(estimator, input_words_file, input_tags_file, output_file):
    output_dir = os.path.dirname(output_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with Path(output_file).open('wb') as f:
        test_inpf = functools.partial(input_fn, input_words_file, input_tags_file)
        golds_gen = generator_fn(input_words_file, input_tags_file)
        preds_gen = estimator.predict(test_inpf)
        for golds, preds in zip(golds_gen, preds_gen):
            ((words, _), (_, _)), tags = golds
            for word, tag, tag_pred in zip(words, tags, preds['tags']):
                f.write(b' '.join([word, tag, tag_pred]) + b'\n')
            f.write(b'\n')


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #if len(sys.argv) > 1:
    #    DATADIR = sys.argv[1]
    #else:
    #    DATADIR = './data/ner_i2b2/source1/input'
    #print('DATADIR:{}'.format(DATADIR))
    

    # Logging
    log_dir = args.output_dir
    Path(log_dir).mkdir(exist_ok=True)
    tf.logging.set_verbosity(logging.INFO)
    handlers = [
        logging.FileHandler('{}/main.log'.format(log_dir)),
        logging.StreamHandler(sys.stdout)
    ]
    logging.getLogger('tensorflow').handlers = handlers
    
    # Params
    params = {
        'dim': 300,
        'dim_chars': 100,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': args.epoch, #25,
        'batch_size': 10, #20,
        'buffer': 10000, #15000,
        'char_lstm_size': 25,
        'lstm_size': 100,
        'words': str(Path(args.data_dir, 'vocab.words.txt')),
        'chars': str(Path(args.data_dir, 'vocab.chars.txt')),
        'tags': str(Path(args.data_dir, 'vocab.tags.txt')),
        'glove': str(Path(args.data_dir, 'glove.npz'))
    }
    with Path('{}/params.json'.format(args.data_dir)).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(args.data_dir, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(args.data_dir, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    
    warm_start_from = None
    if args.warm_start_dir:
        warm_start_from = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=args.warm_start_dir,
            # NOTE: attempted with and without `vars_to_warm_start` set
            # vars_to_warm_start='^(?!.*(RMSProp|global_step))'
            vars_to_warm_start='^(?!.*(crf|dense/bias|dense/kernel))',
        )
        
    estimator = tf.estimator.Estimator(model_fn, '{}'.format(args.model_dir), cfg, params, warm_start_from=warm_start_from)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    #hook = tf.contrib.estimator.stop_if_no_increase_hook(
    hook = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #Path('{}/score'.format(args.output_dir)).mkdir(parents=True, exist_ok=True)
    for name in ['train', 'testa', 'testb']:
        input_words_file = fwords(name)
        input_tags_file = ftags(name)
        output_file = '{}/score/{}.preds.txt'.format(args.output_dir, name)    
        write_predictions(estimator, input_words_file, input_tags_file, output_file)
        
    print('chars_lstm_lstm_crf train done!')
