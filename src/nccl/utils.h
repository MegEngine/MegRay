#pragma once

#include <iostream>

#include "nccl.h"

#include "megray/common.h"

namespace MegRay {

#define NCCL_CHECK(expr)                                \
    do {                                                \
        ncclResult_t result = (expr);                   \
        if (result != ncclSuccess) {                    \
            MEGRAY_ERROR("nccl error [%d]: %s", result, \
                         ncclGetErrorString(result));   \
            return MEGRAY_NCCL_ERR;                     \
        }                                               \
    } while (0);

#define NCCL_ASSERT(expr)                               \
    do {                                                \
        ncclResult_t result = (expr);                   \
        if (result != ncclSuccess) {                    \
            MEGRAY_ERROR("nccl error [%d]: %s", result, \
                         ncclGetErrorString(result));   \
            MEGRAY_THROW("nccl error");                 \
        }                                               \
    } while (0);

ncclDataType_t get_nccl_dtype(const DType dtype);

ncclRedOp_t get_nccl_reduce_op(const ReduceOp red_op);

}  // namespace MegRay
