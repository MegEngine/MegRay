/**
 * \file src/ucx/kernel.cu
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megray/ucx/communicator.h"

namespace MegRay {

template <typename T>
__global__ void reduce_sum_kernel(T* i0, T* i1, T* o, size_t len) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len) {
        o[i] = i0[i] + i1[i];
    }
}

template <typename T>
__global__ void reduce_max_kernel(T* i0, T* i1, T* o, size_t len) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len) {
        o[i] = (i0[i] > i1[i]) ? i0[i] : i1[i];
    }
}

template <typename T>
__global__ void reduce_min_kernel(T* i0, T* i1, T* o, size_t len) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len) {
        o[i] = (i0[i] < i1[i]) ? i0[i] : i1[i];
    }
}

template <typename T>
void reduce_helper(T* i0, T* i1, T* o, size_t len, ReduceOp op,
        cudaStream_t stream) {
    size_t block_dim = 512;
    size_t grid_dim = (len + block_dim - 1) / block_dim;
    switch (op) {
        case MEGRAY_SUM:
            reduce_sum_kernel<T><<<grid_dim, block_dim, 0, stream>>>(i0, i1, o, len);
            break;
        case MEGRAY_MAX:
            reduce_max_kernel<T><<<grid_dim, block_dim, 0, stream>>>(i0, i1, o, len);
            break;
        case MEGRAY_MIN:
            reduce_min_kernel<T><<<grid_dim, block_dim, 0, stream>>>(i0, i1, o, len);
            break;
        default:
            MEGRAY_THROW("unknown reduce op");
    }
}

void reduce(void* i0, void* i1, void* o, size_t len,
        DType dtype, ReduceOp op, cudaStream_t stream) {
    switch (dtype) {
        case MEGRAY_INT8:
            reduce_helper<int8_t>((int8_t*)i0, (int8_t*)i1, (int8_t*)o,
                    len, op, stream);
            break;
        case MEGRAY_UINT8:
            reduce_helper<uint8_t>((uint8_t*)i0, (uint8_t*)i1, (uint8_t*)o,
                    len, op, stream);
            break;
        case MEGRAY_INT32:
            reduce_helper<int32_t>((int32_t*)i0, (int32_t*)i1, (int32_t*)o,
                    len, op, stream);
            break;
        case MEGRAY_UINT32:
            reduce_helper<uint32_t>((uint32_t*)i0, (uint32_t*)i1, (uint32_t*)o,
                    len, op, stream);
            break;
        case MEGRAY_INT64:
            reduce_helper<int64_t>((int64_t*)i0, (int64_t*)i1, (int64_t*)o,
                    len, op, stream);
            break;
        case MEGRAY_UINT64:
            reduce_helper<uint64_t>((uint64_t*)i0, (uint64_t*)i1, (uint64_t*)o,
                    len, op, stream);
            break;
        case MEGRAY_FLOAT32:
            reduce_helper<float>((float*)i0, (float*)i1, (float*)o,
                    len, op, stream);
            break;
        case MEGRAY_FLOAT64:
            reduce_helper<double>((double*)i0, (double*)i1, (double*)o,
                    len, op, stream);
            break;
        default:
            MEGRAY_THROW("unknown dtype");
    }
}

} // namespace MegRay
