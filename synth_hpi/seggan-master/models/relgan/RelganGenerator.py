import numpy as np
import tensorflow as tf

from models.Gan import Gen
from models.relgan.RelganMemory import RelationalMemory
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from utils.ops import *


class Generator(Gen):

    # def __init__(self, temperature, vocab_size, batch_size, seq_len, gen_emb_dim,
    #              mem_slots, head_size, num_heads, hidden_dim, start_token, gpre_lr, grad_clip):

    def __init__(self, batch_size, seq_len, vocab_size, grad_clip, gpre_lr, **kwargs):

        self.grad_clip = grad_clip
        self.x_real = tf.placeholder(
            tf.int32, [batch_size, seq_len], name="x_real")
        
        # batch_size x seq_len x vocab_size
        self.x_real_onehot=tf.one_hot(self.x_real, vocab_size)
        assert self.x_real_onehot.get_shape().as_list() == [
            batch_size, seq_len, vocab_size]

        with tf.variable_scope('generator'):
            self.init_generator(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, **kwargs)
        self.set_pretrain_op(gpre_lr)

    def init_generator(
            self, temperature, vocab_size, batch_size, seq_len, gen_emb_dim, mem_slots,
            head_size, num_heads, hidden_dim, start_token):

        x_real = self.x_real
        start_tokens=tf.constant([start_token] * batch_size, dtype=tf.int32)
        output_size=mem_slots * head_size * num_heads

        # build relation memory module
        g_embeddings=tf.get_variable('g_emb', shape=[vocab_size, gen_emb_dim],
                                       initializer=create_linear_initializer(vocab_size))
        gen_mem=RelationalMemory(
            mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
        g_output_unit=create_output_unit(output_size, vocab_size)

        # initial states
        init_states=gen_mem.initial_state(batch_size)

        # ---------- generate tokens and approximated one-hot results (Adversarial) ---------
        gen_o=tensor_array_ops.TensorArray(
            dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)
        gen_x=tensor_array_ops.TensorArray(
            dtype=tf.int32, size=seq_len, dynamic_size=False, infer_shape=True)
        gen_x_onehot_adv=tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                                                        infer_shape=True)  # generator output (relaxed of gen_x)

        # the generator recurrent module used for adversarial training
        def _gen_recurrence(i, x_t, h_tm1, gen_o, gen_x, gen_x_onehot_adv):
            mem_o_t, h_t=gen_mem(x_t, h_tm1)  # hidden_memory_tuple
            o_t=g_output_unit(mem_o_t)  # batch x vocab, logits not probs
            gumbel_t=add_gumbel(o_t)
            next_token=tf.stop_gradient(
                tf.argmax(gumbel_t, axis=1, output_type=tf.int32))
            next_token_onehot=tf.one_hot(next_token, vocab_size, 1.0, 0.0)

            # one-hot-like, [batch_size x vocab_size]
            x_onehot_appr=tf.nn.softmax(tf.multiply(gumbel_t, temperature))

            # x_tp1 = tf.matmul(x_onehot_appr, g_embeddings)  # approximated embeddings, [batch_size x emb_dim]
            # embeddings, [batch_size x emb_dim]
            x_tp1=tf.nn.embedding_lookup(g_embeddings, next_token)

            gen_o=gen_o.write(i, tf.reduce_sum(tf.multiply(
                next_token_onehot, x_onehot_appr), 1))  # [batch_size], prob
            gen_x=gen_x.write(i, next_token)  # indices, [batch_size]

            gen_x_onehot_adv=gen_x_onehot_adv.write(i, x_onehot_appr)

            return i + 1, x_tp1, h_t, gen_o, gen_x, gen_x_onehot_adv
        # build a graph for outputting sequential tokens
        _, _, _, gen_o, gen_x, gen_x_onehot_adv=control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5: i < seq_len,
            body=_gen_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                       init_states, gen_o, gen_x, gen_x_onehot_adv))

        # batch_size x seq_len
        self.gen_o=tf.transpose(gen_o.stack(), perm=[1, 0])
        # batch_size x seq_len
        self.gen_x=tf.transpose(gen_x.stack(), perm=[1, 0])

        self.gen_x_onehot_adv=tf.transpose(gen_x_onehot_adv.stack(), perm=[
            1, 0, 2])  # batch_size x seq_len x vocab_size

        # ----------- pre-training for generator -----------------
        x_emb=tf.transpose(tf.nn.embedding_lookup(g_embeddings, x_real), perm=[
                             1, 0, 2])  # seq_len x batch_size x emb_dim
        g_predictions=tensor_array_ops.TensorArray(
            dtype=tf.float32, size=seq_len, dynamic_size=False, infer_shape=True)

        ta_emb_x=tensor_array_ops.TensorArray(dtype=tf.float32, size=seq_len)
        ta_emb_x=ta_emb_x.unstack(x_emb)

        # the generator recurrent moddule used for pre-training
        def _pretrain_recurrence(i, x_t, h_tm1, g_predictions):
            mem_o_t, h_t=gen_mem(x_t, h_tm1)
            o_t=g_output_unit(mem_o_t)
            g_predictions=g_predictions.write(
                i, tf.nn.softmax(o_t))  # batch_size x vocab_size
            x_tp1=ta_emb_x.read(i)
            return i + 1, x_tp1, h_t, g_predictions

        # build a graph for outputting sequential tokens
        _, _, _, g_predictions=control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < seq_len,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(g_embeddings, start_tokens),
                       init_states, g_predictions))

        g_predictions=tf.transpose(g_predictions.stack(),
                                     perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

        # pre-training loss
        self.pretrain_loss=-tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(x_real, [-1])), vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(
                    g_predictions, [-1, vocab_size]), 1e-20, 1.0)
            )
        ) / (seq_len * batch_size)

    def set_pretrain_op(self, gpre_lr):
        # pre-training op
        self.vars=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        pretrain_opt=tf.train.AdamOptimizer(gpre_lr, beta1=0.9, beta2=0.999)
        pretrain_grad, _=tf.clip_by_global_norm(tf.gradients(
            self.pretrain_loss, self.vars), self.grad_clip)  # gradient clipping
        self.pretrain_op=pretrain_opt.apply_gradients(
            zip(pretrain_grad, self.vars))

    def set_train_op(self, g_loss, optimizer_name, gadv_lr, global_step, nadv_steps, decay):

        self.loss=g_loss

        if decay:
            gadv_lr=tf.train.exponential_decay(
                gadv_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)

        if optimizer_name == "adam":
            g_optimizer=tf.train.AdamOptimizer(
                gadv_lr, beta1=0.9, beta2=0.999)
        elif optimizer_name == "rmsprop":
            g_optimizer=tf.train.RMSPropOptimizer(gadv_lr)
        else:
            raise AttributeError

        # gradient clipping
        grads, _=tf.clip_by_global_norm(
            tf.gradients(g_loss, self.vars), self.grad_clip)
        self.train_op=g_optimizer.apply_gradients(zip(grads, self.vars))

    def generate(self, sess):
        """gemerate fake smples
        """
        return sess.run(self.gen_x)

    def get_nll(self, sess, x):
        pretrain_loss=sess.run(
            self.pretrain_loss, feed_dict={self.x_real: x}
        )
        return pretrain_loss

    def pretrain_step(self, sess, x):
        """pretrain the generator on step"""
        _, g_loss=sess.run(
            [self.pretrain_op, self.pretrain_loss], feed_dict={self.x_real: x})
        return g_loss

    def train(self, sess, x):
        sess.run(
            self.train_op, feed_dict={self.x_real: x})
