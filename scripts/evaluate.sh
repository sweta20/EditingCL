#!/bin/bash

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

module load cuda/10.1.243
module load cudnn/v7.5.0
root_dir=`dirname $0`

moses_scripts=${root_dir}/mosesdecoder/scripts
metric_path=${root_dir}/readability

model_type=nat
constraint=False
input='none'
evaluate=False
subset=test
name=best
checkpoint=checkpoint_best
use_split=ours-fil
root_exp_dir=experiments
root_out_dir=test_results
ckpd=""
tokenize=False
num_iters=10
eval_cl=False
while getopts "i:j:m:x:s:n:p:u:k:q:cetl" opt; do
	case $opt in
		i)
			eid=$OPTARG ;;
		j)
			did=$OPTARG ;;
		m)
			model_type=$OPTARG ;;
		c)
			constraint=True ;;
		p)
			checkpoint=$OPTARG ;;
		u)
			use_split=$OPTARG ;;
		s)
			subset=$OPTARG ;;
		k)
			ckpd=$OPTARG ;;
		x)
			input=$OPTARG ;;
		n)
			name=$OPTARG ;;
		e)
			evaluate=True ;;
		t)
			tokenize=True ;;
		q)
			num_iters=$OPTARG ;;
		l)
			eval_cl=True ;;
		h)
			echo "Usage: evaluate.sh"
			echo "-i Experiment Run id"
			echo "-j Data Run Id"
			exit 0 ;;

    \?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1 ;;
    :)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1 ;;
	esac
done

exp_dir=${root_exp_dir}/exp-${eid}
test_dir=${root_out_dir}/exp-${eid}
mkdir -p ${test_dir}
model_args=""
if [ $constraint == True ]; then
	out_dir=$test_dir/${subset}-const-${name}
	if [ ${model_type} == nat ]; then
		model_args=" ${model_args} --constrained-decoding  "
	else
		model_args=" ${model_args} --constraints  "
	fi;
else
	out_dir=$test_dir/${subset}-${name}
fi;

# out_dir=${out_dir}-${num_iters}
echo "Writing to $out_dir"
mkdir -p $out_dir

if [ $eval_cl == True ]; then
	data_dir=${root_exp_dir}/exp-${did}/data-bin-$ckpd
else
	data_dir=${root_exp_dir}/exp-${did}/data-bin
fi;

tgt=tgt
src=src

if [ ! -f  $out_dir/output.txt ]; then
	if [ ${model_type} == nat ]; then
		echo "Using iterative refinement"
		if [ $input != 'none' ]; then
			CUDA_LAUNCH_BLOCKING=1 fairseq-interactive          \
                -s $src -t $tgt                \
                $data_dir       \
                --input $input        \
                --task translation_lev                   \
                --path $exp_dir/checkpoints${ckpd}/${checkpoint}.pt          \
                --iter-decode-max-iter ${num_iters}                  \
                --iter-decode-eos-penalty 0                 \
                --print-step                                   \
                --retain-iter-history                           \
                --beam 1                                     \
                --prepend-bos \
                --buffer-size 128  \
                --batch-size 128  \
                --remove-bpe ${model_args}  > $out_dir/output.txt 
           
        else
			fairseq-generate $data_dir \
				--gen-subset ${subset} \
				-s $src -t $tgt \
			    --task translation_lev \
			    --path $exp_dir/checkpoints${ckpd}/${checkpoint}.pt \
			    --iter-decode-max-iter ${num_iters} \
			    --iter-decode-eos-penalty 0         \
                --print-step                                   \
                --retain-iter-history                           \
			    --beam 1 --remove-bpe \
			    --print-step \
			    --batch-size 400  > $out_dir/output.txt 
		fi;
	else
		echo "Using simple decoding"
		if [ $input != 'none' ]; then
			fairseq-interactive -s $src -t $tgt  $data_dir \
			    --input $input --buffer-size 128   \
			    --path $exp_dir/checkpoints${ckpd}/${checkpoint}.pt \
			    --batch-size 128 --beam 5 --remove-bpe  ${model_args}  > $out_dir/output.txt
		else
			fairseq-generate $data_dir \
				-s $src -t $tgt \
				--gen-subset ${subset}  \
			    --path $exp_dir/checkpoints${ckpd}/${checkpoint}.pt \
			    --batch-size 128 --beam 5 --remove-bpe > $out_dir/output.txt
		fi;
	fi;
fi;

if [ $use_split == wmt14ende ]; then
	grep ^H $out_dir/output.txt | cut -f3- > $out_dir/hypotheses.txt
else
	if [ $tokenize == True ]; then
		grep ^H $out_dir/output.txt | cut -f3-  \
									| $moses_scripts/recaser/detruecase.perl 2>/dev/null \
									| $moses_scripts/tokenizer/detokenizer.perl -q -l en 2>/dev/null \
									 > $out_dir/hypotheses.txt
	else     
		grep ^H $out_dir/output.txt | cut -f3-  \
									| $moses_scripts/recaser/detruecase.perl 2>/dev/null > $out_dir/hypotheses.txt

	fi;	
fi;


if [ ${model_type} == 'nat' ]; then
	grep ^I $out_dir/output.txt | cut -f2-  > $out_dir/iters.txt
	grep ^O $out_dir/output.txt | cut -f2-  > $out_dir/ops.txt 
fi;		


if [ $evaluate == True ]; then
	pred=$out_dir/hypotheses.txt

	if [[ $use_split == ours-fil ]]; then
		src=data-simp/$subset.src.nograde
		tgt=data-simp/$subset.tgt
		extra_args=" --grade_file data-simp/$subset.grade.tgt "
	elif [[ $use_split == ours-simp-fil ]]; then
		src=data-simp-fil/$subset.src
		tgt=data-simp-fil/$subset.tgt
	fi; 

	. `dirname $0`/compute_metric.sh

	if [ ${model_type} == 'nat' ]; then
		python $metric_path/ops_count.py --ops_file $out_dir/ops.txt > $out_dir/ops_results.txt
	fi;
fi;
