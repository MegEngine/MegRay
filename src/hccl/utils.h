#pragma once

#include "hccl/hccl_types.h"
#include "megray/common.h"

namespace MegRay {

#define MEGRAY_HCCL_CHECK(expr)                         \
    do {                                                \
        HcclResult result = (expr);                     \
        if (result != HCCL_SUCCESS) {                   \
            MEGRAY_ERROR("hccl error [%d]: %s", result, \
                         HcclGetErrorString(result));   \
            return MEGRAY_HCCL_ERR;                     \
        }                                               \
    } while (0);

#define MEGRAY_HCCL_ASSERT(expr)                        \
    do {                                                \
        HcclResult result = (expr);                     \
        if (result != HCCL_SUCCESS) {                   \
            MEGRAY_ERROR("hccl error [%d]: %s", result, \
                         HcclGetErrorString(result));   \
            MEGRAY_THROW("hccl error");                 \
        }                                               \
    } while (0);

HcclDataType as_hccl_dtype(DType dt);
HcclReduceOp as_hccl_reduce_op(ReduceOp rop);

}  // namespace MegRay
