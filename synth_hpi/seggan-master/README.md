This code was mostly copied from https://github.com/liyzcj/seggan.

Changes includes: creat run_train_gen_seggan.sh to train and generate synthetic texts based on I2B2 2010 and N2C2 2018 History of Present Illness (HPI) section data.

Following is the original README of this package.


## Introduction

This is source code of ``Adversarial Sub-sequence for Text Generation``.

## LICENSE

This code is based on [texygen](https://github.com/geek-ai/Texygen) and [RelGAN](https://github.com/weilinie/RelGAN)

## Requirement

```bash
colorama
numpy>=1.12.1
tensorflow>=1.5.0
scipy>=0.19.0
nltk>=3.2.3
tqdm
```

## Get Started

```bash
λ  python .\main.py --help

       USAGE: .\main.py [flags]
flags:

.\main.py:
  --data: Dataset for real Training
    (default: 'image_coco')
  --gan: <seqgan|leakgan|mle|relgan>: Type of GAN to Training
    (default: 'mle')
  --gpu: The GPU used for training
    (default: '0')
    (an integer)
  --mode: <real|oracle|cfg>: Type of training mode
    (default: 'real')
  --model: Experiment name for LeakGan
    (default: 'test')
  --[no]pretrain: only pretrain, Stop after pretrain!
    (default: 'false')
  --[no]restore: Restore pretrain models for relgan
    (default: 'false')

Try --helpfull to get a list of all flags.
```
