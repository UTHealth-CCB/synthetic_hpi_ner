import numpy as np
import tensorflow as tf
import os


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


def init_sess():
    FLAGS = tf.app.flags.FLAGS
    #++ for specify GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    print(f"Using GPU: {FLAGS.gpu}")
    #++ =========================
    #++ For ignore log
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    #++ =========================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    return sess


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
