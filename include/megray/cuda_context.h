/**
 * \file include/megray/cuda_context.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <memory>

#include "megray/context.h"

#ifdef MEGRAY_WITH_CUDA

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                \
    do {                                                \
        cudaError_t status = (expr);                    \
        if (status != cudaSuccess) {                    \
            MEGRAY_ERROR("cuda error [%d]: %s", status, \
                         cudaGetErrorString(status));   \
            return MEGRAY_CUDA_ERR;                     \
        }                                               \
    } while (0)

#define CUDA_ASSERT(expr)                               \
    do {                                                \
        cudaError_t status = (expr);                    \
        if (status != cudaSuccess) {                    \
            MEGRAY_ERROR("cuda error [%d]: %s", status, \
                         cudaGetErrorString(status));   \
            MEGRAY_THROW("cuda error");                 \
        }                                               \
    } while (0)

namespace MegRay {

class CudaContext : public Context {
public:
    CudaContext(cudaStream_t stream) : m_stream{stream} {}
    static std::shared_ptr<CudaContext> make(cudaStream_t stream) {
        return std::make_shared<CudaContext>(stream);
    }
    ContextType type() const override { return MEGRAY_CTX_CUDA; }
    cudaStream_t get_stream() { return m_stream; }

private:
    cudaStream_t m_stream;
};

}  // namespace MegRay

#endif
