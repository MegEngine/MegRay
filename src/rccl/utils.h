#pragma once

#include <iostream>

#include "rccl.h"

#include "megray/common.h"

namespace MegRay {

#define RCCL_CHECK(expr)                                \
    do {                                                \
        ncclResult_t result = (expr);                   \
        if (result != ncclSuccess) {                    \
            MEGRAY_ERROR("nccl error [%d]: %s", result, \
                         ncclGetErrorString(result));   \
            return MEGRAY_RCCL_ERR;                     \
        }                                               \
    } while (0);

#define RCCL_ASSERT(expr)                               \
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
