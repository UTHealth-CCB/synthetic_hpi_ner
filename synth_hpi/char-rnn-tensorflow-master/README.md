This code was mostly copied from https://github.com/sherjilozair/char-rnn-tensorflow.

Changes includes: create run_gen_charrnn.sh and run_gen_charrnn.sh to train and generate synthetic texts based on I2B2 2010 and N2C2 2018 History of Present Illness (HPI) section data; minor modifications to main.py, model.py, and utils.py.

Following is the original README of this package.

char-rnn-tensorflow

Join the chat at https://gitter.im/char-rnn-tensorflow/Lobby Coverage Status Build Status

Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using Tensorflow.

Inspired from Andrej Karpathy's char-rnn.

Requirements
Tensorflow 1.0
Basic Usage
To train with default parameters on the tinyshakespeare corpus, run python train.py. To access all the parameters use python train.py --help.

To sample from a checkpointed model, python sample.py. Sampling while the learning is still in progress (to check last checkpoint) works only in CPU or using another GPU. To force CPU mode, use export CUDA_VISIBLE_DEVICES="" and unset CUDA_VISIBLE_DEVICES afterward (resp. set CUDA_VISIBLE_DEVICES="" and set CUDA_VISIBLE_DEVICES= on Windows).

To continue training after interruption or to run on more epochs, python train.py --init_from=save

Datasets
You can use any plain text file as input. For example you could download The complete Sherlock Holmes as such:

cd data
mkdir sherlock
cd sherlock
wget https://sherlock-holm.es/stories/plain-text/cnus.txt
mv cnus.txt input.txt
Then start train from the top level directory using python train.py --data_dir=./data/sherlock/

A quick tip to concatenate many small disparate .txt files into one large training file: ls *.txt | xargs -L 1 cat >> input.txt.

Tuning
Tuning your models is kind of a "dark art" at this point. In general:

Start with as much clean input.txt as possible e.g. 50MiB
Start by establishing a baseline using the default settings.
Use tensorboard to compare all of your runs visually to aid in experimenting.
Tweak --rnn_size up somewhat from 128 if you have a lot of input data.
Tweak --num_layers from 2 to 3 but no higher unless you have experience.
Tweak --seq_length up from 50 based on the length of a valid input string (e.g. names are <= 12 characters, sentences may be up to 64 characters, etc). An lstm cell will "remember" for durations longer than this sequence, but the effect falls off for longer character distances.
Finally once you've done all that, only then would I suggest adding some dropout. Start with --output_keep_prob 0.8 and maybe end up with both --input_keep_prob 0.8 --output_keep_prob 0.5 only after exhausting all the above values.
Tensorboard
To visualize training progress, model graphs, and internal state histograms: fire up Tensorboard and point it at your log_dir. E.g.:

$ tensorboard --logdir=./logs/
Then open a browser to http://localhost:6006 or the correct IP/Port specified.

Roadmap
 Add explanatory comments
 Expose more command-line arguments
 Compare accuracy and performance with char-rnn
 More Tensorboard instrumentation
Contributing
Please feel free to:

Leave feedback in the issues
Open a Pull Request
Join the gittr chat
Share your success stories and data sets!
