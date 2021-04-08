#!/bin/bash -eu

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
