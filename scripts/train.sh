#!/bin/bash
# Sweta Agrawal

# set -e
module load cuda/10.1.243
module load cudnn/7.5.0

root_dir=`dirname $0`

moses_scripts=${root_dir}/mosesdecoder/scripts
metric_path=${root_dir}/readability

bpe_num_operations=32000
lang=en

use_split=ours-fil             # ours vs auto-grade vs auto-simple vs ours-fil
model_type=nat
nat_type=editor
noise_type=random_delete_shuffle
suffix=""
restore_file=None
gpus=0,1
model_args=""
architecture=reposition_levenshtein_transformer
update=2
run_n=1
tokenize=False
max_epoch=50
while getopts "i:j:m:u:n:z:s:r:g:a:c:f:e:p:k:tb" opt; do
	case $opt in
		i)
			eid=$OPTARG ;;
		j)
			did=$OPTARG ;;
        s)
            suffix=$OPTARG ;;
		m)
			model_type=$OPTARG ;;
		u)
			use_split=$OPTARG ;;
		n)
			nat_type=$OPTARG ;;
        z)
            noise_type=$OPTARG ;;
        r)
            restore_file=$OPTARG ;;
        g)
            gpus=$OPTARG ;;
        a)
            model_args=$OPTARG ;;
        c)
            architecture=$OPTARG ;;
        p)
            update=$OPTARG ;;
        k)
            run_n=$OPTARG ;;
        e)
            max_epoch=$OPTARG ;;  
        t)
            tokenize=True ;;
     \?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1 ;;
    :)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1 ;;
	esac
done

root_exp_dir=experiments

exp_dir=${root_exp_dir}/exp-${eid}
mkdir -p $exp_dir

tgt=tgt
src=src
data_dir=data
train=$data_dir/train
dev=$data_dir/dev
test=$data_dir/test
src=src 

# Tokenized text in .src and .trg
out_dir=${root_exp_dir}/exp-${did}/data
if [ ! -d ${out_dir}-bin ]; then
    mkdir -p ${out_dir}
    . `dirname $0`/preprocess.sh
fi;


# Use for finetuning
if [ $restore_file != "None" ]; then
    model_args=" ${model_args} --restore-file ${restore_file} --reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader "
fi; 


gpu_ids=$(echo $gpus | sed "s/,/ /g")
gpu_n=$(echo $gpu_ids | wc -w)

echo "eid:$eid, did:$did, model_type=$model_type, nat_type=${nat_type}, data_split=${use_split}, noise_type=${noise_type}, architecture=${architecture}, arguments=${model_args}, ${gpu_ids}, ${gpu_n}" > $exp_dir/config
cat $exp_dir/config


for seed in $(seq 1 $run_n); do
    
    model_dir=${exp_dir}/checkpoints$seed
    mkdir -p ${model_dir}

    if [ ! -f $model_dir/checkpoint_best.pt ]; then

        if [ $model_type == nat ]; then
            CUDA_VISIBLE_DEVICES=$gpus fairseq-train \
            ${out_dir}-bin   \
            -s $src \
            -t $tgt \
            --save-dir ${model_dir}  \
            --ddp-backend=no_c10d \
            --task translation_lev \
            --criterion nat_loss \
            --arch ${architecture} \
            --noise ${noise_type} \
            --share-all-embeddings \
            --optimizer adam --adam-betas '(0.9,0.98)' \
            --lr 0.0005 --lr-scheduler inverse_sqrt \
            --min-lr '1e-09' --warmup-updates 10000 \
            --warmup-init-lr '1e-07' --label-smoothing 0.1 \
            --dropout 0.3 --weight-decay 0.01 \
            --decoder-learned-pos \
            --encoder-learned-pos \
            --update-freq ${update} \
            --max-tokens-valid 4000 \
            --distributed-world-size $gpu_n    \
            --log-format 'simple' --log-interval 100 \
            --fixed-validation-seed 7 \
            --max-tokens 3900 \
            --best-checkpoint-metric ppl \
            --save-interval-updates 1000 \
            --max-update 500000  \
            --max-epoch $max_epoch \
            --fix-batches-to-gpus \
            --keep-last-epochs 10 \
            --seed $seed \
            ${model_args} > ${model_dir}/log
        else
        	CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
        	    ${out_dir}-bin \
                -s $src  -t $tgt \
        	    --arch ${architecture} --share-decoder-input-output-embed \
        	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        	    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
                --attention-dropout 0.1 --weight-decay 0.0001  --seed $seed \
        	    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        	    --max-tokens 4096 --keep-last-epochs 20 \
                --max-epoch 30 --patience 8 --best-checkpoint-metric ppl \
                --save-dir ${model_dir} ${model_args} > ${model_dir}/log  
        fi
    fi
done