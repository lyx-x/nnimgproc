#!/usr/bin/env bash

out=./results/denoising_mlp_keras
mkdir -p ${out}

# Resume training from a checkpoint
python3 -W ignore samples/denoising/denoising_mlp_keras.py \
    --output_dir ${out} \
    --image_dir ${DATA} \
    --max_image_count 1000 \
    --noise gaussian 0.1 \
    --input_patch 9 \
    --output_patch 9 \
    --layers 128 128 \
    --learning_rate 0.001 \
    --minibatch 64 \
    --epochs 10 \
    --training 64000 \
    --validation 1280 \
    --workers 2

# Generate a noisy image
python3 -W ignore samples/denoising/process.py \
    --input data/lena.png \
    --output_folder ${out} \
    --output_image noisy.png \
    --output_meta meta.pkl \
    --noise gaussian 0.1

echo "Before denoising..."
python3 -W ignore samples/util/psnr.py --noisy results/denoising_mlp_keras/noisy.jpg \
    --clean data/lena.png \
    --noisy ${out}/noisy.png

# Denoise
python3 -W ignore samples/denoising/denoising_mlp_keras_eval.py \
    --model ${out} \
    --input ${out}/noisy.png \
    --meta ${out}/meta.pkl \
    --output ${out}/result.png \
    --patch 9 \
    --workers 2

# Evaluate the performance
echo "After denoising..."
python3 samples/util/psnr.py --noisy results/denoising_mlp_keras/noisy.jpg \
    --clean data/lena.png \
    --noisy ${out}/result.png

# Visualizaation
tensorboard --logdir ${out}
