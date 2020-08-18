/**
 * \file src/ucx/reduce.cpp
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

Status UcxCommunicator::reduce(const void* sendbuff, void* recvbuff, size_t len,
        DType dtype, ReduceOp op, uint32_t root, std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // allocate workspace
    size_t size = get_dtype_size(dtype);
    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, 2 * len * size));

    // init lbuffer and rbuffer
    char* lbuffer = (char*)workspace;
    char* rbuffer = (char*)workspace + len * size;
    CUDA_CHECK(cudaMemcpy(rbuffer, sendbuff, len * size, cudaMemcpyDeviceToDevice));

    // offset to make sure virtual_root is 0
    auto virtual_rank = ring_sub(m_rank, root, m_nranks);

    // we need d rounds to reduce data
    size_t d = 0, t = m_nranks - 1;
    while (t > 0) {
        ++d;
        t >>= 1;
    }

    // on each round half nodes send msg to the other half
    // on the 1-st round , node Bxxxxxx1 sends msg to node Bxxxxxx0
    // on the i-th round , node Bxxx1000 sends msg to node Bxxx0000
    // on the last round , node B1000000 sends msg to node B0000000
    int mask = 0;
    for(size_t i = 0; i < d; i++) {
        int bit = 1 << i;
        if ((virtual_rank & mask) == 0) {
            if ((virtual_rank & bit) != 0) {
                auto virtual_dest = virtual_rank ^ bit;
                auto actual_dest = ring_add(virtual_dest, root, m_nranks);
                if (virtual_dest < m_nranks){ // valid dest
                    MEGRAY_CHECK(_send(rbuffer, len * size, actual_dest));
                    MEGRAY_CHECK(_flush());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            } else {
                auto virtual_src = virtual_rank ^ bit;
                auto actual_src = ring_add(virtual_src, root, m_nranks);
                if (virtual_src < m_nranks){ // valid src
                    MEGRAY_CHECK(_recv(lbuffer, len * size, actual_src));
                    MEGRAY_CHECK(_flush());
                    MegRay::reduce(lbuffer, rbuffer, rbuffer, len, dtype, op, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
        }
        mask = mask ^ bit;
    }
    if (m_rank == root) {
        CUDA_CHECK(cudaMemcpy(recvbuff, rbuffer, len * size, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaFree(workspace));
    return MEGRAY_OK;
}

} // namespace MegRay
