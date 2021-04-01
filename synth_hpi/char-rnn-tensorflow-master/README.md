This code was mostly copied from https://github.com/sherjilozair/char-rnn-tensorflow.

Changes includes: create run_gen_charrnn.sh and run_gen_charrnn.sh to train and generate synthetic texts based on I2B2 2010 and N2C2 2018 History of Present Illness (HPI) section data.

Following is the original README of this package.

# char-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM,RNN) for character-level language models in Python using Tensorflow.

## Requirments
- Python 3.6.1
- TensorFlow 1.3.0


## Generate English text
To train:
```
python train.py --input_file data/shakespeare.txt --name shakespeare --num_steps 50 --num_seqs 32 --learning_rate 0.01 --max_steps 20000
```

To sample
```
python sample.py --converter_path shakespeare/converter.pkl --checkpoint_path shakespeare/model/ --max_length 1000
```


## Generate Chinese Poetries

To train
```
python train.py --use_embedding --input_file data/poetry.txt --name poetry --learning_rate 0.005 --num_steps 26 --num_seqs 32 --max_steps 10000
```

To sample

```
python sample.py --use_embedding --converter_path poetry/converter.pkl --checkpoint_path poetry/model/ --max_length 300
```


## Generate Chinese Novels

To train

```
python train.py --use_embedding True --input_file data/novel.txt --num_steps 80 --name novel --learning_rate 0.005 --num_seqs 32 --num_layers 3 --embedding_size 256 --lstm_size 256 --max_steps 1000000
```

To sample

```
python sample.py --converter_path novel/converter.pkl --checkpoint_path  novel/model/ --use_embedding --max_length 2000 --num_layers 3 --lstm_size 256 --embedding_size 256
```


## Generate Chinese Lyrics


To train

```
python train.py --input_file data/jay.txt --num_steps 20 --batch_size 32 --name jay --max_steps 5000 --learning_rate 0.01 --num_layers 3 --use_embedding
```

To sample

```
python sample.py --converter_path jay/converter.pkl --checkpoint_path  jay/model/ --max_length 500 --use_embedding --num_layers 3 --start_string 我知道
```


## Generate Linux Code

To train

```
python train.py --input_file data/linux.txt --num_steps 100 --name linux --learning_rate 0.01 --num_seqs 32 --max_steps 20000
```


To sample

```
python sample.py --converter_path linux/converter.pkl --checkpoint_path  linux/model/ --max_length 1000
```

## Generate Japanese Text

To train

```
python train.py --input_file data/jpn.txt --num_steps 20 --batch_size 32 --name jpn --max_steps 10000 --learning_rate 0.01 --use_embedding
```

To sample

```
python sample.py --converter_path jpn/converter.pkl --checkpoint_path jpn/model/--max_length 1000 --use_embedding
```


## Learn RNNs

 - [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/syllabus.html)
 - [Andrej Karpathy, The Unreasonable Effectiveness of Recurrent Neural Networks, 2015](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
 - [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)


## Acknowledgement

 - [TensorFlow 中 RNN 实现的正确打开方式](https://zhuanlan.zhihu.com/p/28196873)
 - [完全图解 RNN、RNN 变体、Seq2Seq、Attention 机制](https://zhuanlan.zhihu.com/p/28054589)
 - [hzy46/Char-RNN-TensorFlow](https://github.com/hzy46/Char-RNN-TensorFlow)(The codes are all almost from this.I learn a lot from it and implement char-rnn with some changes)
