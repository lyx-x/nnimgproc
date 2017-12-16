#!/usr/bin/env bash

data=/root/data
out=./results/destreaking_cnn_chainer
img=./data/lena.png
mkdir -p ${out}

# Visualization
tensorboard --logdir ${out} &

# Resume training from a checkpoint
python3 samples/destreaking/destreaking_cnn_chainer.py \
    --output_dir ${out} \
    --image_dir ${data} \
    --max_image_count 1000 \
    --device 0 \
    --streak periodic 20 \
    --input_patch 65 \
    --output_patch 65 \
    --learning_rate 0.0001 \
    --minibatch 32 \
    --epochs 10 \
    --training 64000 \
    --validation 1280 \
    --workers 2

# Generate a noisy image
python3 samples/destreaking/process.py \
    --input ${img} \
    --output_folder ${out} \
    --output_image noisy.png \
    --output_meta meta.pkl \
    --streak periodic 20

echo "Before denoising..."
python3 samples/util/psnr.py \
    --clean ${img} \
    --noisy ${out}/noisy.png

# De-streak
python3 samples/destreaking/destreaking_cnn_chainer_eval.py \
    --model ${out} \
    --device 0 \
    --input ${out}/noisy.png \
    --meta ${out}/meta.pkl \
    --output ${out}/result.png \
    --patch 65 \
    --workers 2

# Evaluate the performance
echo "After denoising..."
python3 samples/util/psnr.py \
    --clean ${img} \
    --noisy ${out}/result.png
