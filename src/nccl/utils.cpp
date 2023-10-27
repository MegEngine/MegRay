#include "utils.h"
#include "megray/cuda_context.h"

namespace MegRay {

ncclDataType_t get_nccl_dtype(const DType dtype) {
    switch (dtype) {
        case MEGRAY_INT8:
            return ncclInt8;
        case MEGRAY_UINT8:
            return ncclUint8;
        case MEGRAY_INT32:
            return ncclInt32;
        case MEGRAY_UINT32:
            return ncclUint32;
        case MEGRAY_INT64:
            return ncclInt64;
        case MEGRAY_UINT64:
            return ncclUint64;
        case MEGRAY_FLOAT16:
            return ncclFloat16;
        case MEGRAY_FLOAT32:
            return ncclFloat32;
        case MEGRAY_FLOAT64:
            return ncclFloat64;
        default:
            MEGRAY_THROW("unknown dtype");
    }
}

ncclRedOp_t get_nccl_reduce_op(const ReduceOp red_op) {
    switch (red_op) {
        case MEGRAY_SUM:
            return ncclSum;
        case MEGRAY_MAX:
            return ncclMax;
        case MEGRAY_MIN:
            return ncclMin;
        default:
            MEGRAY_THROW("unknown reduce op");
    }
}

}  // namespace MegRay
