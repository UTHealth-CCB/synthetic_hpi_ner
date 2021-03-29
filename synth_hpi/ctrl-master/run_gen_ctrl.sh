gpu="0"
model_dir=./training_utils/seqlen256_v1.ckpt
generate_num=512000
output_file=../data/synthetic_output_ctrl.txt
python generation.py \
--gpu $gpu \
--model_dir $model_dir \
--generate_num $generate_num \
--sample_num 1 
--output_txt $output_file \
--temperature 0
