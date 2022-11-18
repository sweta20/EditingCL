import torch
from fairseq import libnat_cuda
from fairseq import libnat


# in_tokens=torch.Tensor([[0,  3, 7, 8, 9, 11,  5, 2, 1, 1]], device=torch.device('cpu')).int()
in_tokens=torch.Tensor([[0, 8, 5, 3, 4, 2,   1, 1, 1,1]], device=torch.device('cpu')).int()
out_tokens=torch.Tensor([[0, 4, 3, 5, 2, 1 ,1, 1,1,1]],  device=torch.device('cpu')).int()
padding_idx=1
unk_idx=10

in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

in_tokens_list = [
    [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
]
out_tokens_list = [
    [t for t in s if t != padding_idx]
    for i, s in enumerate(out_tokens.tolist())
]


full_labels = libnat.suggested_ed3_path(
            in_tokens_list, out_tokens_list, padding_idx
        )

print(full_labels)

# #gpu
in_tokens=torch.cuda.IntTensor([[0, 8, 5, 3, 4, 2,   1, 1, 1,1]]).int()
out_tokens=torch.cuda.IntTensor([[0, 4, 3, 5, 2, 1 ,1, 1,1,1]]).int()
padding_idx=1
unk_idx=10
in_masks = in_tokens.ne(padding_idx)
out_masks = out_tokens.ne(padding_idx)

word_reposition_targets = libnat_cuda.generate_reposition_labels(
        in_tokens.int().contiguous(),
        libnat_cuda.advanced_levenshtein_distance(
            in_tokens.int().contiguous(), out_tokens.int().contiguous(),
            in_masks.sum(1).int().contiguous(), out_masks.sum(1).int().contiguous()
        )
    )

print(word_reposition_targets)


