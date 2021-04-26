/**
 * \file src/ucx/all_reduce.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "communicator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "megray/common.h"
#include "megray/debug.h"
#include "megray/server.h"
#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

template <typename T>
void cpu_reduce_kernel(T* dst, T* a, T* b, ReduceOp op, int len) {
    switch (op) {
        case MEGRAY_SUM:
            for (size_t i = 0; i < len; i++) {
                dst[i] = a[i] + b[i];
            }
            break;
        case MEGRAY_MAX:
            for (size_t i = 0; i < len; i++) {
                dst[i] = std::max(a[i], b[i]);
            }
            break;
        case MEGRAY_MIN:
            for (size_t i = 0; i < len; i++) {
                dst[i] = std::min(a[i], b[i]);
            }
            break;
        default:
            MEGRAY_THROW("unsuport op type");
    }
}

void cpu_reduce(void* dst, void* a, void* b, DType dtype, ReduceOp op,
                int len) {
    switch (dtype) {
        case MEGRAY_INT8:
            cpu_reduce_kernel<int8_t>((int8_t*)dst, (int8_t*)a, (int8_t*)b, op,
                                      len);
            break;
        case MEGRAY_UINT8:
            cpu_reduce_kernel<uint8_t>((uint8_t*)dst, (uint8_t*)a, (uint8_t*)b,
                                       op, len);
            break;
        case MEGRAY_INT32:
            cpu_reduce_kernel<int32_t>((int32_t*)dst, (int32_t*)a, (int32_t*)b,
                                       op, len);
            break;
        case MEGRAY_UINT32:
            cpu_reduce_kernel<uint32_t>((uint32_t*)dst, (uint32_t*)a,
                                        (uint32_t*)b, op, len);
            break;
        case MEGRAY_INT64:
            cpu_reduce_kernel<int64_t>((int64_t*)dst, (int64_t*)a, (int64_t*)b,
                                       op, len);
            break;
        case MEGRAY_FLOAT32:
            cpu_reduce_kernel<float>((float*)dst, (float*)a, (float*)b, op,
                                     len);
            break;
        default:
            MEGRAY_THROW("unsuport dtype");
    }
}

Status ShmCommunicator::_shm_all_reduce(const void* sendbuff, void* recvbuff,
                                        size_t len, DType dtype, ReduceOp op,
                                        std::shared_ptr<Context> ctx) {
    // get cuda stream
    size_t k = 2;
    size_t chunks = m_nranks * k;
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    size_t size = get_dtype_size(dtype);
    size_t buff_size = len * size;

    void* shmadd = alloc_shm(buff_size * 2 + m_nranks * sizeof(int));

    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();

    // compute chunk sizes
    size_t quotient = len / chunks;
    size_t remainder = len % chunks;
    std::vector<size_t> chunk_sizes(chunks, quotient);
    for (size_t i = 0; i < remainder; i++) {
        chunk_sizes[i]++;
    }

    // compute chunk offsets
    std::vector<size_t> offsets(chunks, 0);
    for (size_t i = 1; i < chunks; i++) {
        offsets[i] = offsets[i - 1] + chunk_sizes[i - 1] * size;
    }
    size_t tmp_size = 0;
    for (size_t i = 0; i < k; i++) {
        tmp_size += chunk_sizes[m_rank * k + i];
    }
    CUDA_ASSERT(cudaMemcpyAsync((char*)shmadd + offsets[m_rank * k],
                                (char*)sendbuff + offsets[m_rank * k],
                                tmp_size * size, cudaMemcpyDeviceToHost,
                                stream));
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    int* mutex = (int*)((char*)shmadd + buff_size * 2);
    mutex[m_rank] = 0;
    m_client->barrier();

    uint32_t work_rank, next_rank;
    work_rank = ring_add(m_rank * k, k, chunks);
    CUDA_ASSERT(cudaMemcpyAsync((char*)shmadd + buff_size + offsets[work_rank],
                                (char*)sendbuff + offsets[work_rank],
                                chunk_sizes[work_rank] * size,
                                cudaMemcpyDeviceToHost, stream));
    size_t right_rank = ring_add(m_rank, 1, m_nranks);
    volatile int* right_mutex = mutex + right_rank;
    // copy i+1 and reduce ith part
    for (size_t i = k; i < chunks; i++) {
        CUDA_ASSERT(cudaStreamSynchronize(stream));
        while (*right_mutex < i - k)
            ;
        next_rank = ring_add(work_rank, 1, chunks);
        if (i != chunks - 1) {
            CUDA_ASSERT(cudaMemcpyAsync(
                    (char*)shmadd + buff_size + offsets[next_rank],
                    (char*)sendbuff + offsets[next_rank],
                    chunk_sizes[next_rank] * size, cudaMemcpyDeviceToHost,
                    stream));
        }
        void* dst = (void*)((char*)shmadd + offsets[work_rank]);
        void* a = (void*)((char*)shmadd + offsets[work_rank]);
        void* b = (void*)((char*)shmadd + buff_size + offsets[work_rank]);
        cpu_reduce(dst, a, b, dtype, op, chunk_sizes[work_rank]);
        mutex[m_rank] = i - 1;
        work_rank = next_rank;
    }
    _shm_barrier(mutex);
    // copy output and free workspace
    CUDA_ASSERT(cudaMemcpyAsync(recvbuff, shmadd, len * size,
                                cudaMemcpyHostToDevice, stream));
    m_client->barrier();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status ShmCommunicator::all_reduce(const void* sendbuff, void* recvbuff,
                                   size_t len, DType dtype, ReduceOp op,
                                   std::shared_ptr<Context> ctx) {
    // only support in single machine
    if (!m_is_single_machine) {
        MEGRAY_THROW("shm all_reduce only support in single machine now");
    }
    return _shm_all_reduce(sendbuff, recvbuff, len, dtype, op, ctx);
}

}  // namespace MegRay
