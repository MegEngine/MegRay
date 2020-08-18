/**
 * \file src/ucx/all_gather.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "communicator.h"

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status UcxCommunicator::all_gather(const void* sendbuff, void* recvbuff, size_t sendlen,
        DType dtype, std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy local data
    size_t size = get_dtype_size(dtype);
    CUDA_CHECK(cudaMemcpy((char*)recvbuff + m_rank * sendlen * size, sendbuff,
            sendlen * size, cudaMemcpyDeviceToDevice));

    uint32_t r_rank = (m_rank + 1) % m_nranks;
    uint32_t l_rank = (m_rank + m_nranks - 1) % m_nranks;
    char sync_send, sync_recv;

    // ring all gather
    // each round all nodes pass the msg to next node and next node saves a copy
    // i-th part of msg passes from i-th node to i+1 to i+2 and finally to (i-1)-th node
    // after nranks - 1 rounds every node has all msg
    for (size_t i = 0; i < m_nranks - 1; i++) {
        uint32_t send_rank = ring_sub(m_rank, i, m_nranks);
        uint32_t recv_rank = ring_sub(m_rank, i + 1, m_nranks);
        size_t recvlen = sendlen * size;

        // pass (rank - i)-th part to next node
        // recv (rank - i - 1)-th part from previous node
        MEGRAY_CHECK(_send((char*)recvbuff + send_rank * recvlen, recvlen, r_rank));
        MEGRAY_CHECK(_recv((char*)recvbuff + recv_rank * recvlen, recvlen, l_rank));
        MEGRAY_CHECK(_flush());

        // synchronization
        MEGRAY_CHECK(_send(&sync_send, sizeof(char), l_rank));
        MEGRAY_CHECK(_recv(&sync_recv, sizeof(char), r_rank));
        MEGRAY_CHECK(_flush());
    }

    return MEGRAY_OK;
}

} // namespace MegRay
