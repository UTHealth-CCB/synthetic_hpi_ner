gpu="0"
train_step=1000
data_dir=../data
data_name=i2b2_n2c2_hpi
#124M 355M 774M 1558M
model_name=1558M 
checkpoint_dir=checkpoint
run_name=${data_name}-${model_name}-epoch$epoch #mt_sample_ori_epoch$epoch 
nsamples=10000
#./output/output_${data_name}_synth_${model_name}_epoch${epoch}_sample${nsamples}.txt  #$root_dir/output/output_mt_sample_ori_${model_name}_epoch${epoch}_sample${nsamples}.txt 
destination_path=../data/synthetic_output_gpt2.txt
pygenerate=./generate_gpt_2_simple.py

python $pygenerate \
    --gpu $gpu \
    --checkpoint_dir $checkpoint_dir \
    --run_name $run_name \
    --destination_path $destination_path \
    --nsamples $nsamples
