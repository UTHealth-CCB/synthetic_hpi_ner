from pathlib import Path
import functools
import json
import argparse
import os

import tensorflow as tf

from main import parse_fn, model_fn, generator_fn, input_fn
from main import write_predictions

# hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for NER task')

parser.add_argument('--gpu', type=str, default='3', help='available gpu number')
# for transfer learning, data_dir/model_dir may be from source datadir/model_dir,
# and input_words_file/input_tags_file are for the predict file
default_data_root_dir = '/home/jli34/data/Experiments/TransferLearning/data/i2b2_hpi/fold_1' 
parser.add_argument('--input_words_file', type=str, default=default_data_root_dir +'/input/testb.words.txt', help='predict words txt file')
parser.add_argument('--input_tags_file', type=str, default=default_data_root_dir +'/input/testb.words.txt', help='predict tags txt file')
parser.add_argument('--output_file', type=str, default=default_data_root_dir + '/output/epoch25/score/testb.preds-debug.txt', help='output preds file')
parser.add_argument('--data_dir', type=str, default=default_data_root_dir +'/input', help='input datadir') # datadir which contains params/words/chars/tags/glove files for trained model
parser.add_argument('--model_dir', type=str, default=default_data_root_dir + '/model/epoch25', help='model dir')

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params_file = os.path.join(args.data_dir, 'params.json')
    with Path(params_file).open() as f:
        params = json.load(f)

    params['words'] = str(Path(args.data_dir, 'vocab.words.txt'))
    params['chars'] = str(Path(args.data_dir, 'vocab.chars.txt'))
    params['tags'] = str(Path(args.data_dir, 'vocab.tags.txt'))
    params['glove'] = str(Path(args.data_dir, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, args.model_dir, params=params)
    write_predictions(estimator, args.input_words_file, args.input_tags_file, args.output_file)
    #predictions = predict_fn(parse_fn(LINE))
    #print(predictions)
    print('predict done.')
