gpu="0"
name=i2b2_n2c2
converter_path=./$name/converter.pkl
checkpoint_path=./$name/model/
output_file=../data/synthetic_output_charrnn.txt
CUDA_VISIBLE_DEVICES=$gpu python sample.py \
    --converter_path $converter_path \
    --checkpoint_path $checkpoint_path \
    --max_length 3000000 | tee $output_file
