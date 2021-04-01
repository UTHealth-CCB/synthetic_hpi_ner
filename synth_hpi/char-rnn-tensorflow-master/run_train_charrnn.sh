GPU="0"
data_dir=../data
input_file=$data_dir/i2b2_n2c2_hpi.txt
name=i2b2_n2c2
num_steps=100
num_seqs=32
learning_rate=0.001
max_steps=20000
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --input_file $input_file \
    --name $name \
    --num_steps $steps \
    --num_seqs $num_seqs \ 
    --learning_rate $learning_rate \
    --max_steps $max_steps
