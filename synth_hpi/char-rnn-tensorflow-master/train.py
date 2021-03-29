import codecs
import os
from utils import TextReader, batch_generator, pick_top_n
from CharRNN import CharRNN
import tensorflow as tf
import time

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'the name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden layer')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'if use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5,
                      'dropout rate during training process')
tf.flags.DEFINE_string('input_file', None, 'utf-8 encoded input file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps of training')
tf.flags.DEFINE_integer('save_model_every', 200,
                        'save the model every 1000 steps')
tf.flags.DEFINE_integer('log_every', 10, 'log the summaries every 10 steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'the maximum of char number')


def main(_):
    if not os.path.exists(FLAGS.name):
        os.makedirs(FLAGS.name)

    model_path = os.path.join(FLAGS.name, 'model')
    logdir_path = os.path.join(FLAGS.name, 'logdir')
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    Reader = TextReader(text, FLAGS.max_vocab)
    Reader.save_to_file(os.path.join(FLAGS.name, 'converter.pkl'))

    arr = Reader.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            char_rnn = CharRNN(
                num_classes=Reader.vocab_size,
                num_seqs=FLAGS.num_seqs,
                num_steps=FLAGS.num_steps,
                lstm_size=FLAGS.lstm_size,
                num_layers=FLAGS.num_layers,
                learning_rate=FLAGS.learning_rate,
                train_keep_prob=FLAGS.train_keep_prob,
                use_embedding=FLAGS.use_embedding,
                embedding_size=FLAGS.embedding_size)

            # define training procedure
            global_step = tf.Variable(0, trainable=False, name='global_step')
            optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate)
            # use clipping gradients
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(ys=char_rnn.loss, xs=tvars),
                clip_norm=char_rnn.grad_clip)
            train_op = optimizer.apply_gradients(
                grads_and_vars=zip(grads, tvars), global_step=global_step)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            if os.path.exists(model_path):
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                print('model restored!')
            else:
                os.makedirs(model_path)

            new_state = sess.run(char_rnn.initial_state)

            for x, y in g:
                start = time.time()
                feed_dict = {
                    char_rnn.inputs: x,
                    char_rnn.targets: y,
                    char_rnn.keep_prob: FLAGS.train_keep_prob,
                    char_rnn.initial_state: new_state
                }

                _, step, new_state, loss = sess.run([
                    train_op, global_step, char_rnn.final_state, char_rnn.loss
                ], feed_dict)

                end = time.time()
                current_step = tf.train.global_step(sess, global_step)
                if step % FLAGS.log_every == 0:
                    print('step: {}/{}... '.format(step, FLAGS.max_steps),
                          'loss: {:.4f}... '.format(loss),
                          '{:.4f} sec/batch'.format((end - start)))
                if current_step % FLAGS.save_model_every == 0:
                    saver.save(
                        sess,
                        os.path.join(model_path, 'model.ckpt'),
                        global_step=current_step)
                if current_step >= FLAGS.max_steps:
                    break

if __name__ == '__main__':
    tf.app.run()
    print('train done.')