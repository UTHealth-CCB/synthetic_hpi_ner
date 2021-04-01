from abc import abstractmethod, ABCMeta

from utils.utils import init_sess
from utils.text_process import code_to_text, get_tokenlized
import os
import numpy as np
import tensorflow as tf

class Gan(metaclass=ABCMeta):

    def __init__(self):
        self.oracle = None
        self.generator = None
        self.discriminator = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.oracle_data_loader = None
        self.sess = init_sess()
        self.metrics = list()
        self.epoch = 0
        self.log = None
        self.reward = None
        # temp file
        self.oracle_file = None
        self.generator_file = None
        self.test_file = None
        # pathes
        self.output_path = None
        self.save_path = None
        self.summary_path = None
        # dict
        self.wi_dict = None
        self.iw_dict = None

    def set_config(self, config):
        self.__dict__.update(config.dict)

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_data_loader(self, gen_loader, dis_loader, oracle_loader):
        self.gen_data_loader = gen_loader
        self.dis_data_loader = dis_loader
        self.oracle_data_loader = oracle_loader

    def set_sess(self, sess):
        self.sess = sess

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        # in use
        self.epoch = 0
        return

    def evaluate_scores(self):
        from time import time
        log = "epoch:" + str(self.epoch) + '\t'
        scores = list()
        scores.append(self.epoch)
        for metric in self.metrics:
            tic = time()
            score = metric.get_score()
            log += metric.get_name() + ":" + str(score) + '\t'
            toc = time()
            print(f"time elapsed of {metric.get_name()}: {toc - tic:.1f}s")
            scores.append(score)
        print(log)
        return scores

    def evaluate(self):
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        with open(self.log, 'a') as log:
            if self.epoch == 0 or self.epoch == 1:
                head = ["epoch"]
                for metric in self.metrics:
                    head.append(metric.get_name())
                log.write(','.join(head) + '\n')
            scores = self.evaluate_scores()
            log.write(','.join([str(s) for s in scores]) + '\n')
        return scores
    
    def evaluate_real(self):
        self.generate_samples()
        self.get_real_test_file()
        self.evaluate()

    def get_real_test_file(self):
        with open(self.generator_file, 'r') as file:
            codes = get_tokenlized(self.generator_file)
        output = code_to_text(codes=codes, dictionary=self.iw_dict)
        with open(self.test_file, 'w', encoding='utf-8') as outfile:
            outfile.write(output)
        output_file = os.path.join(self.output_path, f"epoch_{self.epoch}.txt")
        with open(output_file, 'w', encoding='utf-8') as of:
            of.write(output)

    def generate_samples(self, oracle=False):
        if oracle:
            generator = self.oracle
            output_file = self.oracle_file
        else:
            generator = self.generator
            output_file = self.generator_file
        # Generate Samples
        generated_samples = []
        for _ in range(int(self.generated_num / self.batch_size)):
            generated_samples.extend(generator.generate(self.sess))
        codes = list()

        with open(output_file, 'w') as fout:
            for sent in generated_samples:
                buffer = ' '.join([str(x) for x in sent]) + '\n'
                fout.write(buffer)

    def pre_train_epoch(self):
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        self.gen_data_loader.reset_pointer()

        for it in range(self.gen_data_loader.num_batch):
            batch = self.gen_data_loader.next_batch()
            g_loss = self.generator.pretrain_step(self.sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)


    def init_real_metric(self):

        from utils.metrics.Nll import Nll
        from utils.metrics.DocEmbSim import DocEmbSim
        from utils.others.Bleu import Bleu
        from utils.metrics.SelfBleu import SelfBleu
        
        if self.nll_gen:
            nll_gen = Nll(self.gen_data_loader, self.generator, self.sess)
            nll_gen.set_name('nll_gen')
            self.add_metric(nll_gen)
        if self.doc_embsim:
            doc_embsim = DocEmbSim(
                self.oracle_file, self.generator_file, self.vocab_size)
            doc_embsim.set_name('doc_embsim')
            self.add_metric(doc_embsim)
        if self.bleu:
            FLAGS = tf.app.flags.FLAGS
            dataset = FLAGS.data
            if dataset == "image_coco":
                real_text = 'data/testdata/test_coco.txt'
            elif dataset == "emnlp_news":
                real_text = 'data/testdata/test_emnlp.txt'
            else:
                raise ValueError
            for i in range(3, 4):
                bleu = Bleu(
                    test_text=self.test_file,
                    real_text=real_text, gram=i)
                bleu.set_name(f"Bleu{i}")
                self.add_metric(bleu)
        if self.selfbleu:
            for i in range(2, 6):
                selfbleu = SelfBleu(test_text=self.test_file, gram=i)
                selfbleu.set_name(f"Selfbleu{i}")
                self.add_metric(selfbleu)

    def save_summary(self):
        # summary writer
        self.sum_writer = tf.summary.FileWriter(
            self.summary_path, self.sess.graph)


    def check_valid(self):
        # TODO
        pass

    @abstractmethod
    def train_oracle(self):
        pass

    def train_cfg(self):
        pass

    def train_real(self):
        pass


class Gen(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def get_nll(self):
        pass

    @abstractmethod
    def pretrain_step(self):
        pass


class Dis(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        pass
