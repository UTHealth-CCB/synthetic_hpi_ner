import tensorflow as tf


class CharRNN(object):
    def __init__(self,
                 num_classes,
                 num_seqs=64,
                 num_steps=50,
                 lstm_size=128,
                 num_layers=2,
                 learning_rate=0.001,
                 grad_clip=5,
                 sample=False,
                 train_keep_prob=0.5,
                 use_embedding=False,
                 embedding_size=128):
        if sample:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        self.build_inputs()
        self.build_lstm()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(
                tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(
                tf.int32,
                shape=(self.num_seqs, self.num_steps),
                name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            if self.use_embedding:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device('/cpu:0'):
                    embedding = tf.get_variable(
                        name='embedding',
                        shape=[self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(
                        embedding, self.inputs)

    def build_lstm(self):
        # 创建单个cell并堆叠多层
        def build_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                cell=lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell([
                build_cell(self.lstm_size, self.keep_prob)
                for _ in range(self.num_layers)
            ])
            self.initial_state = cell.zero_state(
                batch_size=self.num_seqs, dtype=tf.float32)

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=self.lstm_inputs,
                initial_state=self.initial_state)

            seq_output = tf.concat(self.lstm_outputs, axis=1)
            x = tf.reshape(seq_output, shape=[-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(initial_value=tf.truncated_normal(
                    shape=[self.lstm_size, self.num_classes], stddev=1.0))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.nn.xw_plus_b(x, softmax_w, softmax_b)
            self.prediction = tf.nn.softmax(
                logits=self.logits, name='predictions')

        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, shape=self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)
