gpu="0"
save_dir=ckpt_i2b2_n2c2_hpi
output_file=../data/synthetic_output_charrnn.txt
sample=1
sample_num=600000
CUDA_VISIBLE_DEVICES=$gpu python sample.py \
    --save_dir $save_dir \
    --sample $sample \
    -n $sample_num | tee $output_file
     
