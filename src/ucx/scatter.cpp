/**
 * \file src/ucx/scatter.cpp
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

#include "utils.h"

#include "megray/cuda_context.h"

namespace MegRay {

Status UcxCommunicator::scatter(const void* sendbuff, void* recvbuff,
                                size_t recvlen, DType dtype, uint32_t root,
                                std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA,
                  "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // perform send recv
    size_t bytes = recvlen * get_dtype_size(dtype);
    if (m_rank == root) {
        for (size_t r = 0; r < m_nranks; r++) {
            const char* p = (const char*)sendbuff + r * bytes;
            if (r == root) {
                CUDA_CHECK(cudaMemcpy(recvbuff, p, bytes,
                                      cudaMemcpyDeviceToDevice));
            } else {
                MEGRAY_CHECK(_isend(p, bytes, r));
            }
        }
    } else {
        MEGRAY_CHECK(_irecv(recvbuff, bytes, root));
    }
    MEGRAY_CHECK(_flush());
    return MEGRAY_OK;
}

}  // namespace MegRay
