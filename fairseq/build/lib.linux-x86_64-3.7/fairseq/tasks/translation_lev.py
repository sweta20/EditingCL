# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random

import torch

from fairseq.data import noising
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq import utils

@register_task('translation_lev')
class TranslationLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete','random_shuffle', 'random_delete_shuffle', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument('--random-seed', default=1, type=int)
        parser.add_argument('--use-source', default=0, type=int)
        parser.add_argument('--use-const', default=0, type=int)
        parser.add_argument('--refine', default=0, type=int)
        parser.add_argument('--dual-data', default=0, type=int)
        parser.add_argument('--delete-prob', default=0.5, type=float)
        parser.add_argument('--skip-tokens-refine', default=1, type=int, help="skip grade tokens when refining")
        parser.add_argument(
            '--pacing',
            default='no_pacing',
            choices=['no_pacing','root', 'linear'])
        parser.add_argument('--warmup-iters', default=5000, type=int)
        # parser.add_argument('--order', default=0, type=int)
        parser.add_argument('--decay-schedule', default=0.0, type=float)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        # modify madeupword0004 to map to <sep>
        # if self.args.refine:
        #     self.src_dict.add_symbol("<sep>")
        #     self.tgt_dict.add_symbol("<sep>")


        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True, refine=self.args.refine
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_shuffle(target_tokens, p, max_shuffle_distance):
            word_shuffle = noising.WordShuffle(self.tgt_dict)
            target_mask = target_tokens.eq(self.tgt_dict.pad())
            target_length = target_mask.size(1) - target_mask.long().sum(1)
            prev_target_tokens, _ = word_shuffle.noising(
                target_tokens.t().cpu(), target_length.cpu(), max_shuffle_distance)
            prev_target_tokens = prev_target_tokens.to(target_tokens.device).t()
            masks = (target_tokens.clone().sum(dim=1, keepdim=True).float()
                .uniform_(0, 1) < p)
            prev_target_tokens = masks * prev_target_tokens + (~masks) * target_tokens
            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete_shuffle':
            return _random_shuffle(_random_delete(target_tokens), self.args.delete_prob, 3)
        elif self.args.noise == 'random_shuffle':
            return _random_shuffle(target_tokens, self.args.delete_prob, 3)
        elif self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, args):
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            del_reward=getattr(args, 'iter_decode_deletion_reward', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            constrained_decoding=getattr(args, 'constrained_decoding', False),
            hard_constrained_decoding=getattr(args, 'hard_constrained_decoding', False),
            random_seed=getattr(args, 'random_seed', 1))

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()

        if self.args.use_source:
            src = sample['net_input']['src_tokens']
            # skip-tokens-refine : add bos, skip grade tokens, add source, add pad to match length
            src = torch.cat((src[:, 0][:, None], src[:, self.args.skip_tokens_refine + 1:], torch.full_like(src[:, :self.args.skip_tokens_refine], self.tgt_dict.pad())), axis=1)
            prev_target = src
        elif self.args.refine:
            prev_target = sample['net_input']['const_tokens']
        else:
            prev_target = sample['target']

        # path specific training
        if self.args.dual_data:
            if random.random() < 0.5:
                # use reposition -> insertion
                sample['prev_target'] = self.inject_noise(prev_target)
                model.decoder.prob = 0.0
            else:
                # use insertion -> reposition
                sample['prev_target'] = self.inject_noise(sample['target'])
                model.decoder.prob = 1.0
        else:
            sample['prev_target'] = self.inject_noise(prev_target)

        if self.args.decay_schedule != 0.0 :
            model.decoder.oracle_prob = self.args.decay_schedule * model.decoder.oracle_prob

        loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            # Do validation on refinement vs generation task??

            if self.args.use_source:
                src = sample['net_input']['src_tokens']
                # skip-tokens-refine : add bos, skip grade tokens, add source, add pad to match length
                src = torch.cat((src[:, 0][:, None], src[:, self.args.skip_tokens_refine + 1:], torch.full_like(src[:, :self.args.skip_tokens_refine], self.tgt_dict.pad())), axis=1)
                prev_target = src
            elif self.args.refine:
                prev_target = sample['net_input']['const_tokens']
            else:
                prev_target = sample['target']

            if self.args.dual_data:
                if random.random() < 0.5:
                    sample['prev_target'] = self.inject_noise(prev_target)
                    model.decoder.prob = self.args.dual_path_probability
                else:
                    sample['prev_target'] = self.inject_noise(sample['target'])
                    model.decoder.prob = 1.0 - self.args.dual_path_probability
            else:
                sample['prev_target'] = self.inject_noise(prev_target)

            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
