/**
 * \file src/shm/all_to_all.cpp
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
#include "megray/cuda_context.h"

namespace MegRay {

Status ShmCommunicator::_shm_all_to_all(const void* sendbuff, void* recvbuff,
                                        size_t len, DType dtype,
                                        std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    auto size = get_dtype_size(dtype);
    auto buff_size = len * size;
    void* shmadd =
            alloc_shm(buff_size * m_nranks * m_nranks + m_nranks * sizeof(int));
    int* mutex = (int*)((uint8_t*)shmadd + buff_size * m_nranks * m_nranks);
    for (int i = 0; i < m_nranks; i++)
        mutex[i] = 0;
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    m_client->barrier();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (auto i = 0; i < m_nranks; i++) {
        auto offset_vertical = buff_size * i;
        auto offset_split = buff_size * m_nranks * i + m_rank * buff_size;
        CUDA_ASSERT(cudaMemcpyAsync((uint8_t*)shmadd + offset_split,
                                    (uint8_t*)sendbuff + offset_vertical,
                                    buff_size, cudaMemcpyDeviceToHost,
                                    stream_vec[i]));
    }
    // 1  2  3   4  5
    // to assure that
    //          1
    //          2
    //          3
    //          4
    //          5
    // all copy has done but waiting for the other process
    volatile int* rank_mutex = mutex + m_rank;
    barrier_streams_local_process();
    *rank_mutex = 1;
    _shm_barrier_sum((volatile int*)mutex);
    uint8_t* shm_out = (uint8_t*)shmadd + m_rank * m_nranks * buff_size;
    CUDA_ASSERT(cudaMemcpyAsync(recvbuff, shm_out, buff_size * m_nranks,
                                cudaMemcpyHostToDevice, stream));
    m_client->barrier();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status ShmCommunicator::all_to_all(const void* sendbuff, void* recvbuff,
                                   size_t len, DType dtype,
                                   std::shared_ptr<Context> ctx) {
    // only support in single machine
    if (!m_is_single_machine) {
        MEGRAY_THROW("ucx all_reduce only support in single machine now");
    }
    return _shm_all_to_all(sendbuff, recvbuff, len, dtype, ctx);
}

}  // namespace MegRay
