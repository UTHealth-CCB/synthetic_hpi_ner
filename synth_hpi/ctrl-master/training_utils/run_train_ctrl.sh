gpu="0"
model_dir=seqlen256_v1.ckpt/
iterations=256
CUDA_VISIBLE_DEVICES=$gpu python training.py \
    --model_dir $model_dir \
    --iterations $iterations
