/**
 * \file src/cncl/utils.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2023 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "cncl.h"

#include "megray/common.h"

namespace MegRay {

#define MEGRAY_CNCL_CHECK(expr)                         \
    do {                                                \
        cnclResult_t result = (expr);                   \
        if (result != CNCL_RET_SUCCESS) {               \
            MEGRAY_ERROR("cncl error [%d]: %s", result, \
                         cnclGetErrorStr(result));      \
            return MEGRAY_CNCL_ERR;                     \
        }                                               \
    } while (0);

#define MEGRAY_CNCL_ASSERT(expr)                        \
    do {                                                \
        cnclResult_t result = (expr);                   \
        if (result != CNCL_RET_SUCCESS) {               \
            MEGRAY_ERROR("cncl error [%d]: %s", result, \
                         cnclGetErrorStr(result));      \
            MEGRAY_THROW("cncl error");                 \
        }                                               \
    } while (0);

cnclDataType_t get_cncl_dtype(const DType dtype);

cnclReduceOp_t get_cncl_reduce_op(const ReduceOp red_op);

}  // namespace MegRay
