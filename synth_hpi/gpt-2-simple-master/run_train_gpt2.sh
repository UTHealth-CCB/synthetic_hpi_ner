gpu="0"
train_step=1000
data_dir=../data
data_name=i2b2_n2c2_hpi
data_file=${data_dir}/${data_name}.txt
#124M 355M 774M 1558M
model_name=1558M 
pyfinetune=./finetuning.py
checkpoint_dir=checkpoint
run_name=${data_name}-${model_name}-step${train_step}

CUDA_VISIBLE_DEVICES=$gpu python $pyfinetune \
    --model_name $model_name \
    --data_file $data_file  \
    --train_step $train_step \
    --checkpoint_dir $checkpoint_dir \
    --run_name $run_name

