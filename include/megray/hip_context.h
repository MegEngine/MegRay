/**
 * \file include/megray/hip_context.h
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

#ifdef MEGRAY_WITH_HIP

#include <hip/hip_runtime.h>

#define HIP_CHECK(expr)                                \
    do {                                               \
        hipError_t status = (expr);                    \
        if (status != hipSuccess) {                    \
            MEGRAY_ERROR("hip error [%d]: %s", status, \
                         hipGetErrorString(status));   \
            return MEGRAY_HIP_ERR;                     \
        }                                              \
    } while (0)

#define HIP_ASSERT(expr)                               \
    do {                                               \
        hipError_t status = (expr);                    \
        if (status != hipSuccess) {                    \
            MEGRAY_ERROR("hip error [%d]: %s", status, \
                         hipGetErrorString(status));   \
            MEGRAY_THROW("hip error");                 \
        }                                              \
    } while (0)

namespace MegRay {

class HipContext : public Context {
public:
    HipContext(hipStream_t stream) : m_stream{stream} {}
    static std::shared_ptr<HipContext> make(hipStream_t stream) {
        return std::make_shared<HipContext>(stream);
    }
    ContextType type() const override { return MEGRAY_CTX_HIP; }
    hipStream_t get_stream() { return m_stream; }

private:
    hipStream_t m_stream;
};

}  // namespace MegRay

#endif
