#!/bin/bash

# Summarization Experiment

# AR  
bash scripts/train.sh -i 1 -j 1 -m seq2seq -u data-summ

# EDITOR (From Reference) 
bash scripts/train.sh -i 2 -j 1 -m nat -u data-summ 

# Editing Roll-in  -> skip tokens is used to remove grade tokens when using to initialize on the decoder side
bash scripts/train.sh -i 3 -j 1 -m nat -u data-summ -r experiments/exp-2/checkpoints1/checkpoint_best.pt -a " --skip-tokens-refine 0  --use-source 1  --noisy-expert --lr 0.0001 "

# Editing CL
bash scripts/train.sh -i 4 -j 1 -m nat -u ours-fil -r experiments/exp-2/checkpoints1/checkpoint_best.pt -a " --skip-tokens-refine 0 --use-source 1 --noisy-expert --pacing root --lr 0.0001 "
