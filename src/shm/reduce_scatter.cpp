/**
 * \file src/shm/reduce_scatter.cpp
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

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status ShmCommunicator::_shm_reduce_scatter(const void* sendbuff,
                                            void* recvbuff, size_t recvlen,
                                            DType dtype, ReduceOp op,
                                            std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    // void* shmadd = alloc_shm();
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    // one is used for compute
    // one is used for translate
    int k = 2;
    // chunks will split the date to
    // the size of m_nranks * 2
    size_t chunks = m_nranks * k;
    size_t size = get_dtype_size(dtype);
    size_t buff_size = m_nranks * recvlen * size;
    void* shmadd = alloc_shm(buff_size * 2 + m_nranks * sizeof(int));
    size_t quotient = m_nranks * recvlen / chunks;
    size_t remainder = m_nranks * recvlen % chunks;
    std::vector<size_t> chunk_sizes(chunks, quotient);
    for (size_t i = 0; i < remainder; i++) {
        chunk_sizes[i]++;
    }
    // compute chunks offset
    std::vector<size_t> offsets(chunks, 0);
    for (size_t i = 1; i < chunks; i++) {
        offsets[i] = offsets[i - 1] + chunk_sizes[i - 1] * size;
    }
    // the buffer compute + transfer
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
    CUDA_ASSERT(cudaMemcpyAsync(
            recvbuff, (uint8_t*)shmadd + m_rank * recvlen * size,
            recvlen * size, cudaMemcpyHostToDevice, stream));
    m_client->barrier();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}
Status ShmCommunicator::reduce_scatter(const void* sendbuff, void* recvbuff,
                                       size_t recvlen, DType dtype, ReduceOp op,
                                       std::shared_ptr<Context> ctx) {
    // only support in single machine
    if (!m_is_single_machine) {
        MEGRAY_THROW(
                "shm reduce all_reduce only support in single machine now");
    }
    return _shm_reduce_scatter(sendbuff, recvbuff, recvlen, dtype, op, ctx);
}

}  // namespace MegRay
