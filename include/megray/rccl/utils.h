/**
 * \file src/nccl/utils.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <iostream>

#include "nccl.h"

#include "megray/core/common.h"

namespace MegRay {

#define RCCL_CHECK(expr)                                   \
    do {                                                   \
        ncclResult_t result = (expr);                      \
        if (result != ncclSuccess) {                       \
            MEGRAY_ERROR("nccl error [%d]: %s", result,    \
                ncclGetErrorString(result));               \
            return MEGRAY_RCCL_ERR;                        \
        }                                                  \
    } while (0);

#define RCCL_ASSERT(expr)                                  \
    do {                                                   \
        ncclResult_t result = (expr);                      \
        if (result != ncclSuccess) {                       \
            MEGRAY_ERROR("nccl error [%d]: %s", result,    \
                ncclGetErrorString(result));               \
            MEGRAY_THROW("nccl error");                    \
        }                                                  \
    } while (0);

ncclDataType_t get_nccl_dtype(const DType dtype);

ncclRedOp_t get_nccl_reduce_op(const ReduceOp red_op);

} // namespace MegRay
