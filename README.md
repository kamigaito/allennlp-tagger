# A sequential tagger implemented on AllenNLP

This tagger can work in Japanese by default.

## File format

See example files in `dataset`.

## Train

See the example script `train.sh`.
```
### train.sh ###

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
```

## Predict

See the example script `predict.sh`.
```
### predict.sh ###

DATADIR="./dataset"
MODELDIR="./models"
OUTPUTDIR="./output"

epoch=0
for id in `seq 0 40`; do
    if [ -e ${MODELDIR}/save_${id}.save ]; then
        epoch=${id}
    fi
done
echo ${epoch}

python predict.py \
    --input_file  ${DATADIR}/test.txt \
    --output_file  ${OUTPUTDIR}/test.label \
    --tagger_vocab_file ${MODELDIR}/vocab \
    --tagger_param_file ${MODELDIR}/params.pkl \
    --tagger_model_file ${MODELDIR}/save_${epoch}.save \
    --gpuid 0
```

## License

Apache License 2.0
