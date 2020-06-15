/**
 * \file src/ucx/all_to_all.cpp
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

namespace MegRay {

Status UcxCommunicator::all_to_all(const void* sendbuff, void* recvbuff,
        size_t len, DType dtype, std::shared_ptr<Context> ctx) {
    // get cuda stream
    MEGRAY_ASSERT(ctx->type() == MEGRAY_CTX_CUDA, "only cuda context supported");
    cudaStream_t stream = static_cast<CudaContext*>(ctx.get())->get_stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // perform send recv
    size_t bytes = len * get_dtype_size(dtype);
    for (size_t r = 0; r < m_nranks; r++) {
        const char* p = (const char*)sendbuff + r * bytes;
        char* q = (char*)recvbuff + r * bytes;
        if (r == m_rank) {
            CUDA_CHECK(cudaMemcpy(q, p, bytes, cudaMemcpyDeviceToDevice));
        } else {
            MEGRAY_CHECK(_send(p, bytes, r));
            MEGRAY_CHECK(_recv(q, bytes, r));
        }
    }
    MEGRAY_CHECK(_flush());
    return MEGRAY_OK;
}

} // namespace MegRay
