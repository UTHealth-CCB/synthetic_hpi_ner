import getopt
import sys
import os

from colorama import Fore
import tensorflow as tf

from models.leakgan.Leakgan import Leakgan
from models.mle.Mle import Mle
from models.seqgan.Seqgan import Seqgan
from models.relgan.Relgan import Relgan

from utils.config import Config
from utils.text_process import save_dict_file, get_dict_from_vocab

gans = {
    'seqgan': Seqgan,
    'leakgan': Leakgan,
    'mle': Mle,
    'relgan': Relgan,
}
training_mode = {'oracle', 'cfg', 'real'}


def set_gan(gan_name):
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)


def set_training(gan, training_method):
    try:
        if training_method == 'oracle':
            gan_func = gan.train_oracle
        elif training_method == 'cfg':
            gan_func = gan.train_cfg
        elif training_method == 'real':
            gan_func = gan.train_real
        else:
            print(Fore.RED + 'Unsupported training setting: ' +
                  training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' +
              training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def def_flags():
    flags = tf.app.flags
    flags.DEFINE_enum('gan', 'relgan', list(gans.keys()),
                      'Type of GAN to Training')
    flags.DEFINE_enum('mode', 'real', training_mode, 'Type of training mode')
    #flags.DEFINE_string('data', 'i2b2_n2c2',
    #                    'Dataset for real Training')
    flags.DEFINE_string('data_file', 'i2b2_n2c2_hpi.txt',
                        'Dataset for real Training')
    flags.DEFINE_string('vocab_file', 'i2b2_n2c2_hpi.vocab.pkl',
                        'Dataset for real Training')
    flags.DEFINE_boolean('restore', False, 'Restore pretrain models for relgan')
    flags.DEFINE_boolean('pretrain', False, 'only pretrain, Stop after pretrain!')
    flags.DEFINE_string('model', "test", 'Experiment name for LeakGan')
    flags.DEFINE_integer('gpu', 4, 'The GPU used for training')
    return


def main(args):
    FLAGS = tf.app.flags.FLAGS
    gan = set_gan(FLAGS.gan)
    # experiment path
    experiment_path = os.path.join('experiments', FLAGS.model)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    print(f"{Fore.BLUE}Experiment path: {experiment_path}{Fore.RESET}")

    # tempfile
    tmp_path = os.path.join(experiment_path, 'tmp')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    gan.oracle_file = os.path.join(tmp_path, 'oracle.txt')
    gan.generator_file = os.path.join(tmp_path, 'generator.txt')
    if 'output_file' not in FLAGS:
        gan.test_file = os.path.join(tmp_path, 'test_file.txt')        
    else:
        gan.test_file = FLAGS.output_file

    # Log file
    gan.log = os.path.join(
        experiment_path, f'log-{FLAGS.model}.csv')
    if os.path.exists(gan.log) and not FLAGS.restore:
        print(f"{Fore.RED}[Error], Log file {gan.log} exist!{Fore.RESET}")
        exit(-3)

    # Config file
    config_file = os.path.join(experiment_path, 'config.json')
    if not os.path.exists(config_file):
        config_file = os.path.join('models', FLAGS.gan, 'config.json')
        # copy config file
        from shutil import copyfile
        copyfile(config_file, os.path.join(experiment_path, 'config.json'))
        if not os.path.exists(config_file):
            print(f"{Fore.RED}[Error], Config file not exist!{Fore.RESET}")
    print(f"{Fore.BLUE}Using config: {config_file}{Fore.RESET}")
    config = Config(config_file)
    gan.set_config(config)

    # output path
    gan.output_path = os.path.join(experiment_path, 'output')
    if not os.path.exists(gan.output_path):
        os.mkdir(gan.output_path)

    # save path
    gan.save_path = os.path.join(experiment_path, 'ckpts')
    gan.restore = FLAGS.restore
    gan.pretrain = FLAGS.pretrain
    if not os.path.exists(gan.save_path):
        os.mkdir(gan.save_path)

    # summary path
    gan.summary_path = os.path.join(experiment_path, 'summary')
    if not os.path.exists(gan.summary_path):
        os.mkdir(gan.summary_path)

    # preprocess real data
    data_file = FLAGS.data_file #f"data/{FLAGS.data}.txt"
    vocab_file = FLAGS.vocab_file #f"data/vocab/{FLAGS.data}.vocab.pkl"

    if not os.path.exists(vocab_file):
        save_dict_file(data_file, vocab_file)
    gan.wi_dict, gan.iw_dict = get_dict_from_vocab(vocab_file)


    train_f = set_training(gan, FLAGS.mode)
    if FLAGS.mode == 'real':
        train_f(data_file)
    else:
        train_f()
    
    #


if __name__ == '__main__':
    def_flags()
    tf.app.run()
