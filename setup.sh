#!/bin/bash

root_dir=`dirname $0`

module load cuda/10.1.243
module load cudnn/v7.5.0

# === Installing Fairseq 
cd $root_dir
cd fairseq
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
pip install cython
python setup.py build_ext --inplace
python setup.py install --user
pip install --editable .

# === Installing Moses scripts
cd $root_dir
git clone https://github.com/moses-smt/mosesdecoder.git

# === Installing BPE scripts 
cd $root_dir
git clone https://github.com/rsennrich/subword-nmt.git

