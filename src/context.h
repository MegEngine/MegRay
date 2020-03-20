/**
 * \file src/context.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "cuda_runtime.h"

namespace MegRay {

typedef enum {
    MEGRAY_CTX_DEFAULT = 0,
    MEGRAY_CTX_CUDA = 1
} ContextType;

/*!
 * MegRay context is an abstraction of communication contexts (e.g. cuda stream)
 * on different platforms, a context should be passed as a parameter when
 * a communicator operation is called
 */
class Context {
    public:
        Context() = default;

        virtual ContextType type() const {
            return MEGRAY_CTX_DEFAULT;
        }

        static std::shared_ptr<Context> make() {
            return std::make_shared<Context>();
        }
};

/*!
 * CudaContext is a wrapper of cuda stream
 */
class CudaContext : public Context {
    public:
        CudaContext() = delete;

        CudaContext(cudaStream_t stream) : m_stream(stream) {}

        ContextType type() const override {
            return MEGRAY_CTX_CUDA;
        }

        static std::shared_ptr<CudaContext> make(cudaStream_t stream) {
            return std::make_shared<CudaContext>(stream);
        }

        cudaStream_t get_stream() const { return m_stream; }

    private:
        cudaStream_t m_stream;
};

} // namespace MegRay
