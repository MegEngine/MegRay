/**
 * \file src/shm/all_gather.cpp
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

// using shm to do all_gather
//              ---
//    ----  ----   ----   ----
//    when do all_gather the node will map to the shm
//    the send buff will be gatter to shm
//         1        2        3
//         shm     123
Status ShmCommunicator::_shm_all_gather(const void* sendbuff, void* recvbuff,
                                        size_t sendlen, DType dtype,
                                        std::shared_ptr<Context> ctx) {
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    auto size = get_dtype_size(dtype);
    auto buff_size = sendlen * size;
    void* shmadd = alloc_shm(buff_size * m_nranks + m_nranks * sizeof(int));
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    std::vector<size_t> offsets(m_nranks);
    for (auto i = 0; i < m_nranks; i++) {
        offsets[i] = i * buff_size;
    }
    CUDA_ASSERT(cudaMemcpyAsync((uint8_t*)shmadd + offsets[m_rank],
                                (uint8_t*)sendbuff, buff_size,
                                cudaMemcpyDeviceToHost, stream));
    int* mutex = (int*)((uint8_t*)shmadd + buff_size * m_nranks);
    mutex[m_rank] = 0;
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    mutex[m_rank] = 1;
    _shm_barrier_sum(mutex);
    CUDA_ASSERT(cudaMemcpyAsync(recvbuff, shmadd, buff_size * m_nranks,
                                cudaMemcpyHostToDevice, stream));
    m_client->barrier();
    CUDA_ASSERT(cudaStreamSynchronize(stream));
    return MEGRAY_OK;
}

Status ShmCommunicator::all_gather(const void* sendbuff, void* recvbuff,
                                   size_t sendlen, DType dtype,
                                   std::shared_ptr<Context> ctx) {
    // only support in single machine
    if (!m_is_single_machine) {
        MEGRAY_THROW("shm all_gather only support in single machine now");
    }
    return _shm_all_gather(sendbuff, recvbuff, sendlen, dtype, ctx);
}
}  // namespace MegRay
