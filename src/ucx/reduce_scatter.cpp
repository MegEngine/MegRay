/**
 * \file src/ucx/reduce_scatter.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megray/ucx/communicator.h"

#include "megray/ucx/utils.h"

#include "megray/cuda/cuda_context.h"

namespace MegRay {

Status UcxCommunicator::reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvlen,
        DType dtype, ReduceOp op, std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // allocate lbuffer and rbuffer
    size_t size = get_dtype_size(dtype);
    char* lbuffer;
    char* rbuffer;
    CUDA_CHECK(cudaMalloc(&lbuffer, recvlen * size));
    CUDA_CHECK(cudaMalloc(&rbuffer, recvlen * m_nranks * size));

    CUDA_CHECK(cudaMemcpy(rbuffer, sendbuff, recvlen * m_nranks * size, cudaMemcpyDeviceToDevice));

    // pass and add (i-1)-th part from i-th node to i+1 to i+2 finally to (i-1)-th node
    // at last i-th node has the sum of i-th part data
    size_t lrank = ring_sub(m_rank, 1, m_nranks);
    size_t rrank = ring_add(m_rank, 1, m_nranks);
    for (size_t i = 1; i < m_nranks; ++i) {
        size_t send_offset = recvlen * size * ring_sub(m_rank, i, m_nranks);
        size_t recv_offset =
                recvlen * size * ring_sub(m_rank, i + 1, m_nranks);
        MEGRAY_CHECK(_send(rbuffer + send_offset, recvlen * size, rrank));
        MEGRAY_CHECK(_recv(lbuffer, recvlen * size, lrank));
        MEGRAY_CHECK(_flush());
        MegRay::reduce(lbuffer, rbuffer + recv_offset, rbuffer + recv_offset,
                recvlen, dtype, op, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    size_t offset = recvlen * size * m_rank;
    CUDA_CHECK(cudaMemcpy(recvbuff, rbuffer + offset, recvlen * size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(lbuffer));
    CUDA_CHECK(cudaFree(rbuffer));

    return MEGRAY_OK;
}

} // namespace MegRay
