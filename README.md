# EditingCL

This repository contains the code for our ACL2022 paper: [An Imitation Learning Curriculum for Text Editing with Non-Autoregressive Models](https://aclanthology.org/2022.acl-long.520/).

<img width="609" alt="editingcl" src="https://user-images.githubusercontent.com/11375341/202484444-f67901e4-8937-45f2-8d09-6693d6b21ae5.png">


## Running the code

### Setup

Run `bash scripts/setup.sh` to install the libraries and dependencies.

### Data

The data for Abstractive Summarization from [Toutanova et al. (2016)](https://aclanthology.org/D16-1033.pdf) can be found in `data-summ`, which contains 6K short input texts, with upto 5 summaries each.. The Newsela data can be requested from [here](https://newsela.com/data/).

### Training

```bash

AR  
> bash scripts/train.sh -i 1 -j 1 -m seq2seq -u data-summ

EDITOR (From Reference) 
> bash scripts/train.sh -i 2 -j 1 -m nat -u data-summ 

Editing Roll-in 
> bash scripts/train.sh -i 3 -j 1 -m nat -u data-summ -r experiments/exp-2/checkpoints1/checkpoint_best.pt -a " --use-source 1  --noisy-expert --lr 0.0001 "

Editing CL
> bash scripts/train.sh -i 4 -j 1 -m nat -u data-summ -r experiments/exp-2/checkpoints1/checkpoint_best.pt -a " --use-source 1 --noisy-expert --pacing root --lr 0.0001 "

```

The `skip-token-refine` is used to ignore the grade tokens on the decoder side for simplification task and should be set to two to ignore the source and grade tokens.


### Evaluation

Example evaluation configs for the summarization tasks can be found in  `scripts/evaluate_summ.sh` respectively.


## Cite the work

If you make use of the code, models, or algorithm, please cite our paper:

```
@inproceedings{agrawal-carpuat-2022-imitation,
    title = "An Imitation Learning Curriculum for Text Editing with Non-Autoregressive Models",
    author = "Agrawal, Sweta  and
      Carpuat, Marine",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.520",
    doi = "10.18653/v1/2022.acl-long.520",
    pages = "7550--7563",
    abstract = "We propose a framework for training non-autoregressive sequence-to-sequence models for editing tasks, where the original input sequence is iteratively edited to produce the output. We show that the imitation learning algorithms designed to train such models for machine translation introduces mismatches between training and inference that lead to undertraining and poor generalization in editing scenarios. We address this issue with two complementary strategies: 1) a roll-in policy that exposes the model to intermediate training sequences that it is more likely to encounter during inference, 2) a curriculum that presents easy-to-learn edit operations first, gradually increasing the difficulty of training samples as the model becomes competent. We show the efficacy of these strategies on two challenging English editing tasks: controllable text simplification and abstractive summarization. Our approach significantly improves output quality on both tasks and controls output complexity better on the simplification task.",
}
```
