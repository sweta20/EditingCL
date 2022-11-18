/**
* Copyright 2017-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*/

#include "edit_dist.h"
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>      // std::pair

template <typename scalar_t>
__global__ void generate_deletion_label_kernel(
        const scalar_t* __restrict__ source,
        const size_t source_size,
        const size_t operation_size,
        int* __restrict__ operations,
        int* __restrict__ labels) {

    const int index = blockIdx.x;
    const int offset = index * operation_size;
    const int offset_label = index * source_size;

    for (int i = 0; i < source_size; i++) {
        labels[offset_label + i] = 0;
    }

    int k = 0;
    for (int i = 0; i < operation_size; i++) {
        if (operations[offset + i] == 0) {
            break;
        } else if (operations[offset + i] == 1) {
            continue;
        } else if (operations[offset + i] == 2) {
            labels[offset_label + k] = 1;
            k++;
        } else {
            labels[offset_label + k] = 0;
            k++;
        }
    }
}

template <typename scalar_t>
__global__ void generate_insertion_label_kernel(
        const scalar_t* __restrict__ target,
        const size_t target_size,
        const size_t operation_size,
        int* __restrict__ operations,
        int* __restrict__ labels,
        int* __restrict__ masks) {

    const int index = blockIdx.x;
    const int offset = index * operation_size;
    const int offset_label = index * target_size;

    int k = 0;
    int u = 0;
    int m = 0;

    for (int i = 0; i < target_size; i++) {
        labels[offset_label + i] = 0;
        masks[offset_label + i] = 0;
    }

    for (int i = 0; i < operation_size-1; i++) {
        if (operations[offset + i] == 0) {
            break;
        } else if (operations[offset + i] == 2) {
            labels[offset_label + k] = u;
            k++;
            u = 0;
        } else if (operations[offset + i] == 1) {
            masks[offset_label + m] = 1;
            u++; m++;
        } else {
            labels[offset_label + k] = u;
            masks[offset_label + m] = 0;
            k++; m++;
            u = 0;
        }
    }
}

template <typename scalar_t>
__global__ void generate_reposition_label_kernel(
        const scalar_t* __restrict__ source,
        const size_t source_size,
        const size_t operation_size,
        int* __restrict__ operations,
        int* __restrict__ labels) {

    const int index = blockIdx.x;
    const int offset = index * operation_size;
    const int offset_label = index * source_size;

    for (int i = 0; i < source_size; i++) {
        labels[offset_label + i] = 0;
    }

    int k = 0;
    for (int i = 0; i < operation_size; i++) {
        if (operations[offset + i] == 0) {
            break;
        } else if (operations[offset + i] == 1) {
            continue;
        } else if (operations[offset + i] == 2) {
            labels[offset_label + k] = 0;
            k++;
        } else {
            labels[offset_label + k] = operations[offset + i] - 3;
            k++;
        }
    }
}

template <typename scalar_t>
__global__ void levenshtein_distance_kernel(
        const scalar_t* __restrict__ source,
        const scalar_t* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const size_t source_size,
        const size_t target_size,
        int* __restrict__ operations,
        int* __restrict__ errors_curr) {

    const int index = blockIdx.x;
    const int offset = index * (source_size + target_size);
    const int d = index * (source_size + 1) * (target_size + 1);
    const int t = target_size + 1;

    auto err_idx = [d, t](int i, int j) { return d + i * t + j; };
    auto opt_idx = [offset](int k) { return offset + k; };

    const int hyp_len = source_length[index];
    const int ref_len = target_length[index];
    const scalar_t* hyp_begin = source + index * source_size;
    const scalar_t* ref_begin = target + index * target_size;

    // dynamic programming
    for (int i = 0; i <= hyp_len; i++) {
        errors_curr[err_idx(i, 0)] = i;
    }
    for (int j = 0; j <= ref_len; j++) {
        errors_curr[err_idx(0, j)] = j;
    }
    for (int i = 1; i <= hyp_len; i++) {
        for (int j = 1; j <= ref_len; j++) {
            errors_curr[err_idx(i, j)] = min(
                min(
                    errors_curr[err_idx(i-1, j)],
                    errors_curr[err_idx(i, j-1)]
                ) + 1,
                errors_curr[err_idx(i-1, j-1)] + 2 * (
                    *(hyp_begin+i-1) == *(ref_begin+j-1) ? 0 : 1
                )
            );
        }
    }

    // back-tracing
    int i = hyp_len;
    int j = ref_len;
    int o = hyp_len + ref_len;

    for (int k = 0; k < source_size + target_size; k++) {
        operations[opt_idx(k)] = 0;
    }

    while ((i >= 0) && (j >= 0)) {
        if ((i == 0) && (j == 0)) {
        break;
        }

        if ((j > 0) && (errors_curr[err_idx(i, j-1)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 1; j--;  // insertion
        } else if ((i > 0) && (errors_curr[err_idx(i-1, j)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 2; i--;  // deletion
        } else {
            o--; operations[opt_idx(o)] = 3; i--; j--;  // do nothing
        }
    }

    // moving to the left
    for (int k = 0; k < hyp_len + ref_len; k++) {
        if (k + o < hyp_len + ref_len) {
            operations[opt_idx(k)] = operations[opt_idx(k+o)];
        } else {
            operations[opt_idx(k)] = 0;  // padding
        }
    }

}

template <typename scalar_t>
__global__ void faster_levenshtein_distance_kernel(
        const scalar_t* __restrict__ source,
        const scalar_t* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const size_t source_size,
        const size_t target_size,
        int* __restrict__ operations) {

    extern __shared__ short errors[];
    auto errors_curr = errors;

    const int index = blockIdx.x;
    const int offset = index * (source_size + target_size);
    const int t = target_size + 1;

    auto err_idx = [t](int i, int j) { return i * t + j; };
    auto opt_idx = [offset](int k) { return offset + k; };

    const int hyp_len = source_length[index];
    const int ref_len = target_length[index];
    const scalar_t* hyp_begin = source + index * source_size;
    const scalar_t* ref_begin = target + index * target_size;

    // dynamic programming
    for (int i = 0; i <= hyp_len; i++) {
        errors_curr[err_idx(i, 0)] = i;
    }
    for (int j = 0; j <= ref_len; j++) {
        errors_curr[err_idx(0, j)] = j;
    }
    for (int i = 1; i <= hyp_len; i++) {
        for (int j = 1; j <= ref_len; j++) {
            errors_curr[err_idx(i, j)] = min(
                min(
                    errors_curr[err_idx(i-1, j)],
                    errors_curr[err_idx(i, j-1)]
                ) + 1,
                errors_curr[err_idx(i-1, j-1)] + 2 * (
                    *(hyp_begin+i-1) == *(ref_begin+j-1) ? 0 : 1
                )
            );
        }
    }

    // back-tracing
    int i = hyp_len;
    int j = ref_len;
    int o = hyp_len + ref_len;

    for (int k = 0; k < source_size + target_size; k++) {
        operations[opt_idx(k)] = 0;
    }

    while ((i >= 0) && (j >= 0)) {
        if ((i == 0) && (j == 0)) {
        break;
        }

        if ((j > 0) && (errors_curr[err_idx(i, j-1)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 1; j--;  // insertion
        } else if ((i > 0) && (errors_curr[err_idx(i-1, j)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 2; i--;  // deletion
        } else {
            o--; operations[opt_idx(o)] = 3; i--; j--;  // do nothing
        }
    }

    // moving to the left
    for (int k = 0; k < hyp_len + ref_len; k++) {
        if (k + o < hyp_len + ref_len) {
            operations[opt_idx(k)] = operations[opt_idx(k+o)];
        } else {
            operations[opt_idx(k)] = 0;  // padding
        }
    }

}

template <typename scalar_t>
__global__ void advanced_levenshtein_distance_kernel(
        const scalar_t* __restrict__ source,
        const scalar_t* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const size_t source_size,
        const size_t target_size,
        int* __restrict__ target_to_source_map,
        int* __restrict__ operations,
        int* __restrict__ errors_curr) {

    const int index = blockIdx.x;
    const int offset = index * (source_size + target_size);
    const int d = index * (source_size + 1) * (target_size + 1);
    const int t = target_size + 1;

    auto err_idx = [d, t](int i, int j) { return d + i * t + j; };
    auto opt_idx = [offset](int k) { return offset + k; };

    const int hyp_len = source_length[index];
    const int ref_len = target_length[index];
    const int tgt_offset = index * target_size;
    const scalar_t* hyp_begin = source + index * source_size;
    const scalar_t* ref_begin = target + index * target_size;

    // prepare the mapping from target words to source indices
    for (int i = 0; i < ref_len; i++) {
        target_to_source_map[tgt_offset + i] = -1;
        for (int j = 0; j < hyp_len; j++) {
            if (*(ref_begin+i) == *(hyp_begin+j)) {
                target_to_source_map[tgt_offset + i] = j;
                break;
            }
        }
    }

    // dynamic programming
    for (int i = 0; i <= hyp_len; i++) {
        errors_curr[err_idx(i, 0)] = i;
    }
    for (int j = 0; j <= ref_len; j++) {
        errors_curr[err_idx(0, j)] = j;
    }
    for (int i = 1; i <= hyp_len; i++) {
        for (int j = 1; j <= ref_len; j++) {
            errors_curr[err_idx(i, j)] = min(
                min(
                    errors_curr[err_idx(i-1, j)],
                    errors_curr[err_idx(i, j-1)]
                ) + 1,
                errors_curr[err_idx(i-1, j-1)] + 2 * (
                    *(hyp_begin+i-1) == *(ref_begin+j-1) ? 0 : 1
                )
            );
            if (target_to_source_map[tgt_offset+j-1] >= 0) {
                errors_curr[err_idx(i, j)] = min(
                    errors_curr[err_idx(i, j)],
                    errors_curr[err_idx(i-1, j-1)] + 1
                );
            }
        }
    }

    // back-tracing
    int i = hyp_len;
    int j = ref_len;
    int o = hyp_len + ref_len;

    for (int k = 0; k < source_size + target_size; k++) {
        operations[opt_idx(k)] = 0;
    }

    while ((i >= 0) && (j >= 0)) {
        if ((i == 0) && (j == 0)) {
            break;
        }

        if ((j > 0) && (errors_curr[err_idx(i, j-1)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 1; j--;  // insertion
        } else if ((i > 0) && (errors_curr[err_idx(i-1, j)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 2; i--;  // deletion
        } else if ((i > 0) && (j > 0) && (errors_curr[err_idx(i-1, j-1)] < errors_curr[err_idx(i, j)])) {
            o--;
            operations[opt_idx(o)] = 3 + target_to_source_map[tgt_offset+j-1];  // substitution
            i--; j--;
        } else {
            o--; operations[opt_idx(o)] = 3 + i - 1; i--; j--;  // do nothing
        }
    }

    // moving to the left
    for (int k = 0; k < hyp_len + ref_len; k++) {
        if (k + o < hyp_len + ref_len) {
            operations[opt_idx(k)] = operations[opt_idx(k+o)];
        } else {
            operations[opt_idx(k)] = 0;  // padding
        }
    }

}

template <typename scalar_t>
__global__ void faster_advanced_levenshtein_distance_kernel(
        const scalar_t* __restrict__ source,
        const scalar_t* __restrict__ target,
        const int* __restrict__ source_length,
        const int* __restrict__ target_length,
        const size_t source_size,
        const size_t target_size,
        int* __restrict__ target_to_source_map,
        int* __restrict__ operations) {

    extern __shared__ short errors[];
    auto errors_curr = errors;

    const int index = blockIdx.x;
    const int offset = index * (source_size + target_size);
    const int t = target_size + 1;

    auto err_idx = [t](int i, int j) { return i * t + j; };
    auto opt_idx = [offset](int k) { return offset + k; };

    const int hyp_len = source_length[index];
    const int ref_len = target_length[index];
    const int tgt_offset = index * target_size;
    const scalar_t* hyp_begin = source + index * source_size;
    const scalar_t* ref_begin = target + index * target_size;

    // prepare the mapping from target words to source indices
    for (int i = 0; i < ref_len; i++) {
        target_to_source_map[tgt_offset + i] = -1;
        for (int j = 0; j < hyp_len; j++) {
            if (*(ref_begin+i) == *(hyp_begin+j)) {
                target_to_source_map[tgt_offset + i] = j;
                break;
            }
        }
    }

    // dynamic programming
    for (int i = 0; i <= hyp_len; i++) {
        errors_curr[err_idx(i, 0)] = i;
    }
    for (int j = 0; j <= ref_len; j++) {
        errors_curr[err_idx(0, j)] = j;
    }
    for (int i = 1; i <= hyp_len; i++) {
        for (int j = 1; j <= ref_len; j++) {
            errors_curr[err_idx(i, j)] = min(
                min(
                    errors_curr[err_idx(i-1, j)],
                    errors_curr[err_idx(i, j-1)]
                ) + 1,
                errors_curr[err_idx(i-1, j-1)] + 2 * (
                    *(hyp_begin+i-1) == *(ref_begin+j-1) ? 0 : 1
                )
            );
            if (target_to_source_map[tgt_offset+j-1] >= 0) {
                errors_curr[err_idx(i, j)] = min(
                    errors_curr[err_idx(i, j)],
                    errors_curr[err_idx(i-1, j-1)] + 1
                );
            }
        }
    }

    // back-tracing
    int i = hyp_len;
    int j = ref_len;
    int o = hyp_len + ref_len;

    for (int k = 0; k < source_size + target_size; k++) {
        operations[opt_idx(k)] = 0;
    }

    while ((i >= 0) && (j >= 0)) {
        if ((i == 0) && (j == 0)) {
            break;
        }

        if ((j > 0) && (errors_curr[err_idx(i, j-1)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 1; j--;  // insertion
        } else if ((i > 0) && (errors_curr[err_idx(i-1, j)] < errors_curr[err_idx(i, j)])) {
            o--; operations[opt_idx(o)] = 2; i--;  // deletion
        } else if ((i > 0) && (j > 0) && (errors_curr[err_idx(i-1, j-1)] < errors_curr[err_idx(i, j)])) {
            o--;
            operations[opt_idx(o)] = 3 + target_to_source_map[tgt_offset+j-1];  // substitution
            i--; j--;
        } else {
            o--; operations[opt_idx(o)] = 3 + i - 1; i--; j--;  // do nothing
        }
    }

    // moving to the left
    for (int k = 0; k < hyp_len + ref_len; k++) {
        if (k + o < hyp_len + ref_len) {
            operations[opt_idx(k)] = operations[opt_idx(k+o)];
        } else {
            operations[opt_idx(k)] = 0;  // padding
        }
    }

}


torch::Tensor GenerateDeletionLabelCuda(
        torch::Tensor source,
        torch::Tensor operations) {

    const auto batch_size = source.size(0);
    at::TensorOptions options(source.device());
    options = options.dtype(at::ScalarType::Int);
    auto labels = torch::empty({batch_size, source.size(1)}, options);
    auto stream = at::cuda::getCurrentCUDAStream(source.device().index());

    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "generate_deletion_labels", ([&] {
        generate_deletion_label_kernel<scalar_t><<<batch_size, 1, 0, stream>>>(
            source.data<scalar_t>(),
            source.size(1),
            operations.size(1),
            operations.data<int>(),
            labels.data<int>());
    }));

    return labels;
}

std::pair<torch::Tensor, torch::Tensor> GenerateInsertionLabelCuda(
    torch::Tensor target,
    torch::Tensor operations) {

    const auto batch_size = target.size(0);
    at::TensorOptions options(target.device());
    options = options.dtype(at::ScalarType::Int);
    auto labels = torch::empty({batch_size, target.size(1)}, options);
    auto masks  = torch::empty({batch_size, target.size(1)}, options);
    auto stream = at::cuda::getCurrentCUDAStream(target.device().index());

    AT_DISPATCH_ALL_TYPES(target.scalar_type(), "generate_insertion_labels", ([&] {
        generate_insertion_label_kernel<scalar_t><<<batch_size, 1, 0, stream>>>(
            target.data<scalar_t>(),
            target.size(1),
            operations.size(1),
            operations.data<int>(),
            labels.data<int>(),
            masks.data<int>());
    }));

    return std::make_pair(labels, masks);
}

torch::Tensor GenerateRepositionLabelCuda(
        torch::Tensor source,
        torch::Tensor operations) {

    const auto batch_size = source.size(0);
    at::TensorOptions options(source.device());
    options = options.dtype(at::ScalarType::Int);
    auto labels = torch::empty({batch_size, source.size(1)}, options);
    auto stream = at::cuda::getCurrentCUDAStream(source.device().index());

    AT_DISPATCH_ALL_TYPES(source.scalar_type(), "generate_reposition_labels", ([&] {
        generate_reposition_label_kernel<scalar_t><<<batch_size, 1, 0, stream>>>(
            source.data<scalar_t>(),
            source.size(1),
            operations.size(1),
            operations.data<int>(),
            labels.data<int>());
    }));

    return labels;
}

torch::Tensor LevenshteinDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length) {

    const auto batch_size = source.size(0);
    const auto shared_size = (source.size(1) + 1) * (target.size(1) + 1) * sizeof(short);
    
    at::TensorOptions options(source.device());
    options = options.dtype(at::ScalarType::Int);
    auto operations = torch::empty({batch_size, source.size(1) + target.size(1)}, options);
    auto stream = at::cuda::getCurrentCUDAStream(source.device().index());

    if (shared_size > 40000) {
        auto distances = torch::empty({batch_size, (source.size(1) + 1) * (target.size(1) + 1)}, options);
        AT_DISPATCH_ALL_TYPES(source.scalar_type(), "levenshtein_distance", ([&] {
            levenshtein_distance_kernel<scalar_t><<<batch_size, 1, 0, stream>>>(
                source.data<scalar_t>(),
                target.data<scalar_t>(),
                source_length.data<int>(),
                target_length.data<int>(),
                source.size(1),
                target.size(1),
                operations.data<int>(),
                distances.data<int>());
        }));
    } else {
        AT_DISPATCH_ALL_TYPES(source.scalar_type(), "faster_levenshtein_distance", ([&] {
            faster_levenshtein_distance_kernel<scalar_t><<<batch_size, 1, shared_size, stream>>>(
                source.data<scalar_t>(),
                target.data<scalar_t>(),
                source_length.data<int>(),
                target_length.data<int>(),
                source.size(1),
                target.size(1),
                operations.data<int>());
        }));
    }

    return operations;
}

torch::Tensor AdvancedLevenshteinDistanceCuda(
        torch::Tensor source,
        torch::Tensor target,
        torch::Tensor source_length,
        torch::Tensor target_length) {

    const auto batch_size = source.size(0);
    const auto shared_size = (source.size(1) + 1) * (target.size(1) + 1) * sizeof(short);
    
    at::TensorOptions options(source.device());
    options = options.dtype(at::ScalarType::Int);
    auto target_to_source_map = torch::empty({batch_size, target.size(1)}, options);
    auto operations = torch::empty({batch_size, source.size(1) + target.size(1)}, options);
    auto stream = at::cuda::getCurrentCUDAStream(source.device().index());

    if (shared_size > 40000) {
        auto distances = torch::empty({batch_size, (source.size(1) + 1) * (target.size(1) + 1)}, options);
        AT_DISPATCH_ALL_TYPES(source.scalar_type(), "advanced_levenshtein_distance", ([&] {
            advanced_levenshtein_distance_kernel<scalar_t><<<batch_size, 1, 0, stream>>>(
                source.data<scalar_t>(),
                target.data<scalar_t>(),
                source_length.data<int>(),
                target_length.data<int>(),
                source.size(1),
                target.size(1),
                target_to_source_map.data<int>(),
                operations.data<int>(),
                distances.data<int>());
        }));
    } else {
        AT_DISPATCH_ALL_TYPES(source.scalar_type(), "faster_advanced_levenshtein_distance", ([&] {
            faster_advanced_levenshtein_distance_kernel<scalar_t><<<batch_size, 1, shared_size, stream>>>(
                source.data<scalar_t>(),
                target.data<scalar_t>(),
                source_length.data<int>(),
                target_length.data<int>(),
                source.size(1),
                target.size(1),
                target_to_source_map.data<int>(),
                operations.data<int>());
        }));
    }

    return operations;
}
