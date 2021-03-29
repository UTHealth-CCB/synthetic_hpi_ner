gpu="0"
data_file=../data/i2b2_n2c2_hpi.txt
vocab_file=../data/i2b2_n2c2_hpi.vocab.pkl
output_file=../data/synthetic_output_seggan.txt
gan=relgan
mode=real

python main.py \
    --gpu $gpu
    --data_file $data_file \
    --vocab_file $vocab_file \
    --output_file $output_file \
    --gan $gan \
    --mode $mode