GPU="0"
input_file=../data/i2b2_n2c2_hpi.txt
temp_dir=./temp
save_dir=ckpt_i2b2_n2c2_hpi
seq_length=100
batch_size=32
learning_rate=0.001
num_epochs=100
model=lstm
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --input_file $input_file \
    --temp_dir $temp_dir \
    --save_dir $save_dir \
    --seq_length $seq_length \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_epochs $num_epochs \
    --model=$model

