#!/bin/bash -eu

DATADIR="./dataset"
MODELDIR="./models"

if [ -d ${MODELDIR} ]; then
    rm -rf ${MODELDIR}
fi
mkdir -p ${MODELDIR}

python train.py \
    --train_file ${DATADIR}/train.csv \
    --valid_file ${DATADIR}/dev.csv \
    --model_dir ${MODELDIR} \
    --dim_emb 768 \
    --dim_hid 256 \
    --dropout 0.2 \
    --epochs 10 \
    --batch_size 1 \
    --use_amp \
    --num_gradient_accumulation_steps 1 \
    --num_enc_layers 2 \
    --bert_name "cl-tohoku/bert-base-japanese-whole-word-masking" \
    --with_bert \
    --bert_max_len 512 \
    --min_freq 2 \
    --gpuid 0
