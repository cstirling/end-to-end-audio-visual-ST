#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch # chainer or pytorch
ngpu=2          # number of gpus ("0" uses cpu, otherwise use gpu)
nj=16           # numebr of parallel jobs for decoding
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume=         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/multimodal_train.yaml
decode_config=conf/tuning/decode_pytorch_transformer.yaml

# decoding parameter
trans_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ST models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ST models will be averaged.
                             # if false, the last `n_average` ST models will be averaged.

# pre-training related
# asr_model=/home/jingzhao/espnet/egs/how2/st1/exp/train.pt_tc_pytorch_train_pytorch_transformer_short_long_bpe8000_specaug_asrtrans_mttrans/results/model.val5.avg.best
asr_model=/home/jingzhao/espnet/egs/how2/st1/exp/train.pt_tc_pytorch_multimodal_all_add/decode_dev5.pt_decode_pytorch_transformer.100.loss.true/model.val5.avg.best
mt_model=

# preprocessing related
src_case=lc.rm
tgt_case=tc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
# how2=/export/a13/kduh/mtdata/how2/how2-300h-v1
how2=/home/data/how2/how2-300h-v1
# how2-300h-v1
#  |_ data/
#  |_ features/
#    |_ fbank_pitch_181516/

# bpemode (unigram or bpe)
nbpe=8000
bpemode=bpe

# exp tag
tag="multimodal_all_add_vbs_blank" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train.pt
train_dev=val.pt
trans_set="dev5.pt"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}

dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${tgt_case}.txt
nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${tgt_case}.txt
bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${tgt_case}

if [ -z ${tag} ]; then
    expname=${train_set}_${tgt_case}_${backend}_$(basename ${train_config%.*})_${bpemode}${nbpe}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
    if [ -n "${asr_model}" ]; then
        expname=${expname}_asrtrans
    fi
    if [ -n "${mt_model}" ]; then
        expname=${expname}_mttrans
    fi
else
    expname=${train_set}_${tgt_case}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

echo $tag
echo "stage 4: Network Training"

env CUDA_VISIBLE_DEVICES=2,6 ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    st_train.py \
    --config ${train_config} \
    --preprocess-conf ${preprocess_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --debugmode ${debugmode} \
    --dict ${dict} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --seed ${seed} \
    --verbose ${verbose} \
    --resume ${resume} \
    --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${tgt_case}.json \
    --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${tgt_case}.json \
    --enc-init ${asr_model} \
    --dec-init ${mt_model} \
    --multimodal