/**
 * \file src/ucx/all_reduce.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "communicator.h"

#include <vector>

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status UcxCommunicator::all_reduce(const void* sendbuff, void* recvbuff,
                                   size_t len, DType dtype, ReduceOp op,
                                   std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // compute chunk sizes
    size_t quotient = len / m_nranks;
    size_t remainder = len % m_nranks;
    std::vector<size_t> chunk_sizes(m_nranks, quotient);
    for (size_t i = 0; i < remainder; i++) {
        chunk_sizes[i]++;
    }

    // allocate workspace for recv, chunk_0 is the largest
    size_t size = get_dtype_size(dtype);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, chunk_sizes[0] * size));
    CUDA_CHECK(cudaMemcpy(recvbuff, sendbuff, len * size,
                          cudaMemcpyDeviceToDevice));

    // compute chunk offsets
    std::vector<size_t> offsets(m_nranks, 0);
    for (size_t i = 1; i < m_nranks; i++) {
        offsets[i] = offsets[i - 1] + chunk_sizes[i - 1] * size;
    }

    uint32_t r_rank = (m_rank + 1) % m_nranks;
    uint32_t l_rank = (m_rank + m_nranks - 1) % m_nranks;
    char sync_send, sync_recv;

    // step 1: all reduce chunks
    // split data with n chunks , reduce i-th chunk data at (i-1)-th node
    // pass and add i-th chunk from i-th node to i+1 to i+2 finally to (i-1)-th
    // node at last i-th node has the sum of (i+1)-th chunk data
    for (uint32_t i = 0; i < m_nranks - 1; i++) {
        uint32_t send_chunk = ring_sub(m_rank, i, m_nranks);
        uint32_t recv_chunk = ring_sub(m_rank, i + 1, m_nranks);

        size_t send_offset = offsets[send_chunk];
        size_t recv_offset = offsets[recv_chunk];

        MEGRAY_CHECK(_send((char*)recvbuff + send_offset,
                           chunk_sizes[send_chunk] * size, r_rank));
        MEGRAY_CHECK(_recv((char*)workspace, chunk_sizes[recv_chunk] * size,
                           l_rank));
        MEGRAY_CHECK(_flush());

        MegRay::reduce((char*)recvbuff + recv_offset, (char*)workspace,
                       (char*)recvbuff + recv_offset, chunk_sizes[recv_chunk],
                       dtype, op, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MEGRAY_CHECK(_send(&sync_send, sizeof(char), l_rank));
        MEGRAY_CHECK(_recv(&sync_recv, sizeof(char), r_rank));
        MEGRAY_CHECK(_flush());
    }

    // step 2:  ring allgather
    // each round all nodes pass the msg to next node and next node saves it
    // i-th part of msg passes from i-th node to i+1 to i+2 and finaly to
    // (i-1)-th node after n-1 rounds every node has all msg
    for (uint32_t i = 0; i < m_nranks - 1; i++) {
        uint32_t send_chunk = ring_sub(m_rank + 1, i, m_nranks);
        uint32_t recv_chunk = ring_sub(m_rank, i, m_nranks);

        MEGRAY_CHECK(_send((char*)recvbuff + offsets[send_chunk],
                           chunk_sizes[send_chunk] * size, r_rank));
        MEGRAY_CHECK(_recv((char*)recvbuff + offsets[recv_chunk],
                           chunk_sizes[recv_chunk] * size, l_rank));
        MEGRAY_CHECK(_flush());

        MEGRAY_CHECK(_send(&sync_send, sizeof(char), l_rank));
        MEGRAY_CHECK(_recv(&sync_recv, sizeof(char), r_rank));
        MEGRAY_CHECK(_flush());
    }

    // copy output and free workspace
    CUDA_CHECK(cudaMemcpy(recvbuff, recvbuff, len * size,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(workspace));

    return MEGRAY_OK;
}

}  // namespace MegRay
