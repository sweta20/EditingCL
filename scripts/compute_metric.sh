#!/bin/bash

metric_path=/fs/clip-scratch/sweagraw/Editor/readability

echo "Source: $src, Target: $tgt, Pred: $pred, prefix: $prefix"

# BLEU
if [ ! -f $out_dir/bleu$prefix.log ]; then 
	echo " * Computing bleu* ..." 
	cat $pred | sacrebleu $tgt > $out_dir/bleu$prefix.log
	cat $pred | sacrebleu $src >> $out_dir/bleu$prefix.log
fi;
cat $out_dir/bleu$prefix.log

# SARI
if [ ! -f $out_dir/sari$prefix.log ]; then 
	echo " * Computing SARI* ..." 
	python3 $metric_path/SARI.py \
		--src_file $src \
		--ref_file $tgt \
		--out_file $pred \
		 > $out_dir/sari$prefix.log
fi;
cat $out_dir/sari$prefix.log

# ARI
if [ ! -f $out_dir/ari$prefix.log ]; then 
	echo " * Computing Grade level stats* ..." 
	python $metric_path/compute_grade_stats.py --ref_file $tgt --pred_file $pred ${extra_args} > $out_dir/ari$prefix.log
fi;
cat $out_dir/ari$prefix.log