# -*- coding: utf-8 -*-

import os
import sys
import requests
import getopt

import gpt_2_simple as gpt2

default_gpu = '10'
default_model_name = '774M' #124M 355M 774M 1558M
default_data_name = 'i2b2_n2c2'
default_data_file = 'data/{}.txt'.format(default_data_name)
default_train_step = 2
default_checkpoint_dir = 'checkpoint' 
default_run_name = '{}-{}-{}'.format(default_data_name, default_model_name, default_train_step)

def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, 'h',['gpu=', 'model_name=', 'data_file=', 'train_step=', 'checkpoint_dir=', 'run_name='])
        opt_arg = dict(opts)
        if ('-h' in opt_arg.keys()):
            print('usage: python {} --gpu --model_name --data_file --train_step --checkpoint_dir --run_name'.format(__file__))
            print('       python {} --gpu 0 --model_name <gpt-2 model_name: 124M|355M|774M|1542M> --data_file <data/ner_i2b2.txt: your_data_location> --train_step <1000: max number of training steps> --checkpoint_dir <checkpoint: output checkpoint dir> --run_name <run1: output checkpoint subfolder>'.format(__file__))
            sys.exit(0)
        if '--gpu' not in opt_arg:
            opt_arg['--gpu'] = default_gpu
        if '--model_name' not in opt_arg:
            opt_arg['--model_name'] = default_model_name
        if '--data_file' not in opt_arg:
            opt_arg['--data_file'] = default_data_file
        if '--train_step' not in opt_arg:
            opt_arg['--train_step'] = default_train_step
        if '--checkpoint_dir' not in opt_arg:
            opt_arg['--checkpoint_dir'] = default_checkpoint_dir
        if '--run_name' not in opt_arg:
            opt_arg['--run_name'] = default_run_name            
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    return opt_arg

if __name__ == "__main__":   
    opt_arg = parse_cmd(sys.argv[1:])
    gpu = opt_arg['--gpu']
    model_name = opt_arg['--model_name']
    data_file = opt_arg['--data_file']
    train_step = int(opt_arg['--train_step'])
    checkpoint_dir = opt_arg['--checkpoint_dir']
    run_name = opt_arg['--run_name']
    print('args: {}'.format(opt_arg))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    if not os.path.isdir(os.path.join("models", model_name)):
    	print(f"Downloading {model_name} model...")
    	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
    
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  dataset=data_file,
                  model_name=model_name,
                  checkpoint_dir=checkpoint_dir,
                  run_name=run_name,
                  steps=train_step)   # steps is max number of training steps
    print('finetuning gpt 2 simple done.')