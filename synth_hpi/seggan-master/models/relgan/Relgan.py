import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
from colorama import Fore

from models.Gan import Gan
from models.relgan.RelganDataLoader import RealDataLoader
from models.relgan.RelganDataLoader import OracleDataLoader
from models.relgan.RelganDiscriminator import Discriminator
from models.relgan.RelganGenerator import Generator

from utils.ops import get_losses
from utils.utils import *
from utils.text_process import *
from utils.others.OracleLstm import OracleLstm


class Relgan(Gan):

    def __init__(self):
        super().__init__()
    
    def init_oracle_training(self):
        oracle_model = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.gen_emb_dim,
                                  hidden_dim=self.hidden_dim, sequence_length=self.seq_len,
                                  start_token=self.start_token)
        self.set_oracle(oracle_model)

        # temperature variable
        self.temperature = tf.Variable(1., trainable=False, name='temperature')

        generator = Generator(
            temperature=self.temperature, vocab_size=self.vocab_size, batch_size=self.batch_size,
            seq_len=self.seq_len, gen_emb_dim=self.gen_emb_dim, mem_slots=self.mem_slots,
            head_size=self.head_size, num_heads=self.num_heads, hidden_dim=self.hidden_dim,
            start_token=self.start_token, gpre_lr=self.gpre_lr, grad_clip=self.grad_clip)
        self.set_generator(generator)

        discriminator = Discriminator(
            batch_size=self.batch_size, seq_len=self.seq_len, vocab_size=self.vocab_size,
            dis_emb_dim=self.dis_emb_dim, num_rep=self.num_rep, sn=self.sn, grad_clip=self.grad_clip,
            splited_steps=self.splited_steps
        )
        self.set_discriminator(discriminator)


        oracle_loader = OracleDataLoader(self.batch_size, self.seq_len)
        gen_loader = OracleDataLoader(self.batch_size, self.seq_len)

        self.set_data_loader(gen_loader=gen_loader, dis_loader=None, oracle_loader=oracle_loader)

        # Global step
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_op = self.global_step.assign_add(1)

        # Get losses
        log_pg, g_loss, d_loss = get_losses(
            generator, discriminator, self.gan_type)

        # Set Train ops
        generator.set_train_op(
            g_loss, self.optimizer_name, self.gadv_lr, global_step=self.global_step,
            nadv_steps=self.nadv_steps, decay=self.decay)
        discriminator.set_train_op(
            d_loss, self.optimizer_name, self.d_lr, global_step=self.global_step,
            nadv_steps=self.nadv_steps, decay=self.decay
        )

        # Temperature placeholder
        self.temp_var = tf.placeholder(tf.float32)
        self.update_temperature_op = self.temperature.assign(self.temp_var)

    def init_oracle_metric(self):
        from utils.metrics.Nll import Nll

        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)
    
    def evaluate_oracle(self):
        self.generate_samples()
        self.evaluate()
    
    def train_oracle(self):

        self.init_oracle_training()
        self.init_oracle_metric()

        self.sess.run(tf.global_variables_initializer())

        self.generate_samples(oracle=True)
        self.gen_data_loader.create_batches(self.oracle_file)

        # Saver
        saver_variables = tf.global_variables()
        saver = tf.train.Saver(saver_variables)

        # summary writer
        self.save_summary()

        # restore 
        if self.restore:
            restore_from = tf.train.latest_checkpoint(self.save_path)
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            self.epoch = self.npre_epochs
        else:
            print('start pre-train Relgan:')
            for epoch in range(self.npre_epochs // self.ntest_pre):
                self.evaluate_oracle()
                for _ in tqdm(range(self.ntest_pre), ncols=50):
                    g_pretrain_loss_np = self.pre_train_epoch()
                    self.add_epoch()

                # save pre_train
            saver.save(self.sess, os.path.join(self.save_path, 'pre_train'))

        # stop after pretrain
        if self.pretrain:
            self.evaluate_oracle()
            exit()
            
        print('start adversarial:')
        for _ in range(self.nadv_steps):

            niter = self.sess.run(self.global_step)
            tic = time.time()
            # adversarial training
            for _ in range(self.gsteps):
                self.generator.train(
                    self.sess, self.gen_data_loader.random_batch())
            for _ in range(self.dsteps):
                self.sess.run(
                    self.discriminator.train_op,
                    feed_dict={self.generator.x_real: self.gen_data_loader.random_batch()})

            toc = time.time()

            # temperature
            temp_var_np = get_fixed_temperature(self.temper, niter, self.nadv_steps, self.adapt)
            self.sess.run(
                self.update_temperature_op, 
                feed_dict={self.temp_var: temp_var_np})

            feed = {self.generator.x_real: self.gen_data_loader.random_batch()}
            g_loss_np, d_loss_np = self.sess.run(
                [self.generator.loss, self.discriminator.loss], feed_dict=feed)
            # update global step
            self.sess.run(self.global_step_op)

            # print(f"Epoch: {niter} G-loss: {g_loss_np:.4f} D-loss: {d_loss_np:.4f}  Time: {toc-tic:.1f}s")

            if np.mod(niter, self.ntest) == 0:
                self.evaluate_oracle()
            self.add_epoch()

    def init_real_training(self, data_loc):

        self.seq_len, self.vocab_size = text_precess(data_loc)

        # temperature variable
        self.temperature = tf.Variable(1., trainable=False, name='temperature')

        generator = Generator(
            temperature=self.temperature, vocab_size=self.vocab_size, batch_size=self.batch_size,
            seq_len=self.seq_len, gen_emb_dim=self.gen_emb_dim, mem_slots=self.mem_slots,
            head_size=self.head_size, num_heads=self.num_heads, hidden_dim=self.hidden_dim,
            start_token=self.start_token, gpre_lr=self.gpre_lr, grad_clip=self.grad_clip)
        self.set_generator(generator)

        discriminator = Discriminator(
            batch_size=self.batch_size, seq_len=self.seq_len, vocab_size=self.vocab_size,
            dis_emb_dim=self.dis_emb_dim, num_rep=self.num_rep, sn=self.sn, grad_clip=self.grad_clip,
            splited_steps=self.splited_steps
        )
        self.set_discriminator(discriminator)

        # Global step
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_op = self.global_step.assign_add(1)

        # Get losses
        log_pg, g_loss, d_loss = get_losses(
            generator, discriminator, self.gan_type)

        # Set Train ops
        generator.set_train_op(
            g_loss, self.optimizer_name, self.gadv_lr, global_step=self.global_step,
            nadv_steps=self.nadv_steps, decay=self.decay)
        discriminator.set_train_op(
            d_loss, self.optimizer_name, self.d_lr, global_step=self.global_step,
            nadv_steps=self.nadv_steps, decay=self.decay
        )

        # Temperature placeholder
        self.temp_var = tf.placeholder(tf.float32)
        self.update_temperature_op = self.temperature.assign(self.temp_var)


        # dataloader
        gen_dataloader = RealDataLoader(
            batch_size=self.batch_size, seq_length=self.seq_len,
        )
        self.set_data_loader(gen_loader=gen_dataloader,
                             dis_loader=None, oracle_loader=None)
        tokens = get_tokenlized(data_loc)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, self.wi_dict, self.seq_len))
    
    def init_real_metric(self):
        
        from utils.metrics.Scalar import Scalar
        # temperature
        t = Scalar(self.sess, self.temperature, "Temperature")
        self.add_metric(t)
        super().init_real_metric()

    def train_real(self, data_loc=None):
        self.init_real_training(data_loc)
        self.init_real_metric()

        self.gen_data_loader.create_batches(self.oracle_file)

        self.sess.run(tf.global_variables_initializer())

        # Saver
        saver_variables = tf.global_variables()
        saver = tf.train.Saver(saver_variables)

        # summary writer
        self.save_summary()

        # restore 
        if self.restore:
            restore_from = tf.train.latest_checkpoint(self.save_path)
            saver.restore(self.sess, restore_from)
            print(f"{Fore.BLUE}Restore from : {restore_from}{Fore.RESET}")
            self.epoch = self.npre_epochs
        else:
            print('start pre-train Relgan:')
            for epoch in range(self.npre_epochs // self.ntest_pre):
                self.evaluate_real()
                for _ in tqdm(range(self.ntest_pre), ncols=50):
                    g_pretrain_loss_np = self.pre_train_epoch()
                    self.add_epoch()

                # save pre_train
            saver.save(self.sess, os.path.join(self.save_path, 'pre_train'))

        # stop after pretrain
        if self.pretrain:
            self.evaluate_real()
            exit()
            
        print('start adversarial:')
        for _ in range(self.nadv_steps):

            niter = self.sess.run(self.global_step)
            tic = time.time()
            # adversarial training
            for _ in range(self.gsteps):
                self.generator.train(
                    self.sess, self.gen_data_loader.random_batch())
            for _ in range(self.dsteps):
                self.sess.run(
                    self.discriminator.train_op,
                    feed_dict={self.generator.x_real: self.gen_data_loader.random_batch()})

            toc = time.time()

            # temperature
            temp_var_np = get_fixed_temperature(self.temper, niter, self.nadv_steps, self.adapt)
            self.sess.run(
                self.update_temperature_op, 
                feed_dict={self.temp_var: temp_var_np})

            feed = {self.generator.x_real: self.gen_data_loader.random_batch()}
            g_loss_np, d_loss_np = self.sess.run(
                [self.generator.loss, self.discriminator.loss], feed_dict=feed)
            # update global step
            self.sess.run(self.global_step_op)

            # print(f"Epoch: {niter} G-loss: {g_loss_np:.4f} D-loss: {d_loss_np:.4f}  Time: {toc-tic:.1f}s")

            if np.mod(niter, self.ntest) == 0:
                self.evaluate_real()
            self.add_epoch()


# A function to set up different temperature control policies
def get_fixed_temperature(temper, i, N, adapt):
    if adapt == 'no':
        temper_var_np = temper  # no increase
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * \
            np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 +
                                            np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1)**2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")

    return temper_var_np
