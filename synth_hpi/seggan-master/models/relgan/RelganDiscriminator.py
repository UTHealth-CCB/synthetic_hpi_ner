import tensorflow as tf
import numpy as np

from models.Gan import Dis
from utils.ops import create_linear_initializer, conv2d, highway, linear


class Discriminator(Dis):

    def __init__(self, batch_size, seq_len, vocab_size, dis_emb_dim, num_rep, sn, grad_clip, splited_steps):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.num_rep = num_rep
        self.sn = sn
        self.grad_clip = grad_clip
        self.get_logits = tf.make_template('discriminator', self.logits)
        self.splited_steps = splited_steps

    def logits(self, x_onehot):
        batch_size = self.batch_size
        seq_len = self.seq_len
        vocab_size = self.vocab_size
        dis_emb_dim = self.dis_emb_dim
        num_rep = self.num_rep
        sn = self.sn

        # get the embedding dimension for each presentation
        emb_dim_single = int(dis_emb_dim / num_rep)
        assert isinstance(emb_dim_single, int) and emb_dim_single > 0

        filter_sizes = [2, 3, 4, 5]
        num_filters = [300, 300, 300, 300]
        dropout_keep_prob = 0.75

        d_embeddings = tf.get_variable('d_emb', shape=[vocab_size, dis_emb_dim],
                                       initializer=create_linear_initializer(vocab_size))
        input_x_re = tf.reshape(x_onehot, [-1, vocab_size])
        emb_x_re = tf.matmul(input_x_re, d_embeddings)
        # batch_size x seq_len x dis_emb_dim
        emb_x = tf.reshape(emb_x_re, [batch_size, seq_len, dis_emb_dim])

        # Mycode : split sentences
        with tf.name_scope("spliter"):
            splited_out = []
            for length in self.splited_steps:
                W = np.zeros([1, seq_len, 1])
                W[0, 0:length, 0] = 1
                W = tf.constant(W, dtype=tf.float32, name=f"W{length}")
                result = emb_x * W
                splited_out.append(result)
            
            emb_x_split = tf.concat(splited_out, 0)

        # batch_size x seq_len x dis_emb_dim x 1
        emb_x_expanded = tf.expand_dims(emb_x_split, -1)
        # print('shape of emb_x_expanded: {}'.format(
        #     emb_x_expanded.get_shape().as_list()))

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            conv = conv2d(emb_x_expanded, num_filter, k_h=filter_size, k_w=emb_dim_single,
                          d_h=1, d_w=emb_dim_single, sn=sn, stddev=None, padding='VALID',
                          scope="conv-%s" % filter_size)  # batch_size x (seq_len-k_h+1) x num_rep x num_filter
            out = tf.nn.relu(conv, name="relu")
            pooled = tf.nn.max_pool(out, ksize=[1, seq_len - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1], padding='VALID',
                                    name="pool")  # batch_size x 1 x num_rep x num_filter
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(num_filters)
        # batch_size x 1 x num_rep x num_filters_total
        h_pool = tf.concat(pooled_outputs, 3)
        # print('shape of h_pool: {}'.format(h_pool.get_shape().as_list()))
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add highway
        # (batch_size*num_rep) x num_filters_total
        h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
        h_drop = tf.nn.dropout(h_highway, dropout_keep_prob, name='dropout')

        # fc
        fc_out = linear(h_drop, output_size=100,
                        use_bias=True, sn=sn, scope='fc')
        logits = linear(fc_out, output_size=1,
                        use_bias=True, sn=sn, scope='logits')
        logits = tf.squeeze(logits, -1)  # batch_size*num_rep
        return logits

    def predict(self):
        pass

    def set_train_op(self, d_loss, optimizer_name, d_lr, global_step, nadv_steps, decay):
        
        self.loss = d_loss
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        if decay:
            d_lr = tf.train.exponential_decay(
                d_lr, global_step=global_step, decay_steps=nadv_steps, decay_rate=0.1)
        
        if optimizer_name == "adam":
            d_optimizer = tf.train.AdamOptimizer(d_lr, beta1=0.9, beta2=0.999)
        elif optimizer_name == "rmsprop":
            d_optimizer = tf.train.RMSPropOptimizer(d_lr)
        else:
            raise AttributeError
            
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(d_loss, d_vars), self.grad_clip)
        self.train_op = d_optimizer.apply_gradients(zip(grads, d_vars))
