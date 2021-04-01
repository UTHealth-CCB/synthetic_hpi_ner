# -*- coding: utf-8 -*-

import gpt_2_simple as gpt2
import os
import sys
import requests
import getopt

data_name = 'i2b2_n2c2'
model_name = '1558M' #774M #1558M #124M 355M 774M 1558M
train_step = 1000
default_gpu = '10'
default_checkpoint_dir = 'checkpoint' 
default_run_name = '{}-{}-epoch{}'.format(data_name, model_name, train_step)
default_destination_path = 'output/output_{}_{}_{}.txt'.format(data_name, model_name, train_step) #i2b2-n2c2-7744M-1000.txt'
default_nsamples = 1000

def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, 'h',['gpu=', 'checkpoint_dir=', 'run_name=', 'destination_path=', 'nsamples='])
        opt_arg = dict(opts)
        if ('-h' in opt_arg.keys()):
            print('usage: python {} --gpu --model_name --data_file --train_step --checkpoint_dir --run_name --destination_path --nsamples'.format(__file__))
            print('       python {} --gpu 0 --model_name <gpt-2 model_name: 124M|355M|774M|1542M> --data_file <data/ner_i2b2.txt: your_data_location> --train_step <1000: max number of training steps> --checkpoint_dir <checkpoint: output checkpoint dir> --run_name <run1: output checkpoint subfolder> --destination_path <output/output.txt output text file> --nsamples <1 output number of samples>'.format(__file__))
            sys.exit(0)
        if '--gpu' not in opt_arg:
            opt_arg['--gpu'] = default_gpu
        if '--checkpoint_dir' not in opt_arg:
            opt_arg['--checkpoint_dir'] = default_checkpoint_dir
        if '--run_name' not in opt_arg:
            opt_arg['--run_name'] = default_run_name
        if '--destination_path' not in opt_arg:
            opt_arg['--destination_path'] = default_destination_path
        if '--nsamples' not in opt_arg:
            opt_arg['--nsamples'] = default_nsamples
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    return opt_arg

if __name__ == "__main__":   
    opt_arg = parse_cmd(sys.argv[1:])
    gpu = opt_arg['--gpu']
    checkpoint_dir = opt_arg['--checkpoint_dir']
    run_name = opt_arg['--run_name']
    destination_path = opt_arg['--destination_path']
    nsamples = int(opt_arg['--nsamples'])
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    #print(checkpoint_dir + ":" + run_name + ":" + destination_path + ":" + str(nsamples))
    print('args: {}'.format(opt_arg))    
    
    sess = gpt2.start_tf_sess()
    print(checkpoint_dir + "/" + run_name)
    gpt2.load_gpt2(sess, checkpoint_dir=checkpoint_dir, run_name=run_name)
    # generate some text
    gpt2.generate_to_file(sess, checkpoint_dir=checkpoint_dir, run_name=run_name, destination_path=destination_path, nsamples=nsamples)
    print('generate_gpt_2_simple done.')