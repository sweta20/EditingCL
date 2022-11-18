#!/bin/bash

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

root_dir=`dirname $0`

moses_scripts=${root_dir}/mosesdecoder/scripts
metric_path=${root_dir}/readability
bpe_scripts_path=${root_dir}/subword-nmt/subword_nmt

src=data-summ/test.src
tgt=data-summ/test.tgt

# detokenized src text
$moses_scripts/recaser/truecase.perl \
                -model experiments/exp-1/data/tc                       \
                < $src                    \
                > experiments/exp-1/data/test.src.tc


python $bpe_scripts_path/apply_bpe.py \
                --codes experiments/exp-1/data/bpe               \
                < experiments/exp-1/data/test.src.tc    \
                > experiments/exp-1/data/test.src.tc.bpe

# k -> experiment id
for k in 1; do
        # seq2seq
        bash scripts/evaluate.sh -i 1 -j 1  -k $k -n $k-_best -s test -m seqseq -x experiments/exp-1/data/test.src.tc.bpe -p checkpoint_best
        python $metric_path/SARI.py --src_file $src --ref_file $tgt --out_file test_results/exp-1/test-$k-_best/hypotheses.txt > test_results/exp-1/test-$k-_best/sari.log
        cat test_results/exp-1/test-$k-_best/sari.log


        # editor
        bash scripts/evaluate.sh -i 2 -j 1 -k $k -n $k-_best -s test -m nat -x experiments/exp-1/data/test.src.tc.bpe -p checkpoint_best
        python $metric_path/SARI.py --src_file $src --ref_file $tgt --out_file test_results/exp-2/test-$k-_best/hypotheses.txt > test_results/exp-2/test-$k-_best/sari.log
        cat test_results/exp-2/test-$k-_best/sari.log

        # editor zeroshot
        bash scripts/evaluate.sh -i 2 -j 1 -k $k -n $k-_best -s test -m nat -c -x experiments/exp-1/data/test.const.src -p checkpoint_best
        python $metric_path/SARI.py --src_file $src --ref_file $tgt --out_file test_results/exp-2/test-const-$k-_best/hypotheses.txt > test_results/exp-2/test-const-$k-_best/sari.log
        cat test_results/exp-2/test-const-$k-_best/sari.log

        # noisy expert
        bash scripts/evaluate.sh -i 3 -j 1 -k $k -n $k-_best -s test -m nat -c -x experiments/exp-1/data/test.const.src -p checkpoint_best
        python $metric_path/SARI.py --src_file $src --ref_file $tgt --out_file test_results/exp-3/test-const-$k-_best/hypotheses.txt > test_results/exp-3/test-const-$k-_best/sari.log
        cat test_results/exp-3/test-const-$k-_best/sari.log

        # necl
        bash scripts/evaluate.sh -i 4 -j 1 -k $k -n $k-_best -s test -m nat -c -x experiments/exp-1/data/test.const.src -p checkpoint_best
        python $metric_path/SARI.py --src_file $src --ref_file $tgt --out_file test_results/exp-4/test-const-$k-_best/hypotheses.txt > test_results/exp-4/test-const-$k-_best/sari.log
        cat test_results/exp-4/test-const-$k-_best/sari.log

done;
