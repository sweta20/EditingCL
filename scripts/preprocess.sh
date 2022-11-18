#!/bin/bash


# if [ $tokenize == True ]; then
#     tsrc=$src.nograde
#     for data in train dev; do
#         for type in $tsrc $tgt; do
#             echo "Tokenizing   ${!data}.$type  "
#             cat ${!data}.$type |  $moses_scripts/tokenizer/normalize-punctuation.perl -l en | \
#             $moses_scripts/tokenizer/tokenizer.perl -threads 8 -a -l en >  ${!data}.tok.$type ;
#             if [ $type == $tsrc ]; then
#                 # add target grade after tokenization
#                 paste ${!data}.grade.tgt ${!data}.tok.$type > ${!data}.tok.$src
#             fi;
#         done;
#     done;
#     inf=".tok"
# else
#     inf=""
# fi;

# # Applying truecasing
# tc_model=${out_dir}/tc
# # tc_model=experiments/exp-1/data/tc

# if [ ! -f $tc_model ]; then
#     echo " * Training truecaser using $train.* ..."
#     cat $train$inf.$src $train$inf.$tgt > ${out_dir}/train.tmp
#     $moses_scripts/recaser/train-truecaser.perl \
#         -corpus ${out_dir}/train.tmp          \
#         -model $tc_model
#     rm ${out_dir}/train.tmp
# fi;

# for data in train dev; do
#     for type in $src $tgt; do
#         if [ -f $tc_model ] && [ ! -f ${out_dir}/${data}.tc.$type  ]; then
#             echo " * True-casing ${!data}.$type ..."
#             $moses_scripts/recaser/truecase.perl \
#                 -model $tc_model                       \
#                 < ${!data}$inf.$type                       \
#                 > ${out_dir}/${data}.tc.$type 
#         fi;
#     done;
# done;

# # Byte Pair Encoding (BPE)
# bpe_model=${out_dir}/bpe
# # bpe_model=experiments/exp-1/data/bpe

# if [ ! -f $bpe_model ]; then
#     echo 'Learning BPE'
#     cat ${out_dir}/train.tc.$src  ${out_dir}/train.tc.$tgt \
#         | python $bpe_scripts_path/learn_bpe.py \
#             -s ${bpe_num_operations} \
#             > $bpe_model
# fi;

# for data in train dev; do
#     for type in $src $tgt; do
#         if [ -f $bpe_model ] && [ ! -f ${out_dir}/${data}.tc.bpe.$type   ]; then
#             echo " * Applying BPE to $data.tc.$type ..."
#             python $bpe_scripts_path/apply_bpe.py \
#                 --codes $bpe_model                  \
#                 < ${out_dir}/${data}.tc.$type    \
#                 > ${out_dir}/${data}.tc.bpe.$type 
#         fi;
#     done;
# done;

if [ ! -d $exp_dir/data-bin ]; then
    fairseq-preprocess --source-lang $src --target-lang $tgt \
        --trainpref $out_dir/train.tc.bpe --validpref $out_dir/dev.tc.bpe \
        --joined-dictionary \
        --destdir $exp_dir/data-bin \
        --workers 20

    # fairseq-preprocess --source-lang $src --target-lang $tgt \
    #     --trainpref $out_dir/train.tc.bpe --validpref $out_dir/dev.tc.bpe \
    #     --srcdict experiments/exp-5/data-bin/dict.src.txt  \
    #     --tgtdict experiments/exp-5/data-bin/dict.tgt.txt  \
    #     --destdir $exp_dir/data-bin \
    #     --workers 20
fi;