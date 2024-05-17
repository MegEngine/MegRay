#include "utils.h"

namespace MegRay {

HcclDataType as_hccl_dtype(DType dt) {
    switch (dt) {
        case DType::MEGRAY_INT8:
            return HCCL_DATA_TYPE_INT8;
        case DType::MEGRAY_UINT8:
            return HCCL_DATA_TYPE_UINT8;
        case DType::MEGRAY_INT32:
            return HCCL_DATA_TYPE_INT32;
        case DType::MEGRAY_UINT32:
            return HCCL_DATA_TYPE_UINT32;
        case DType::MEGRAY_INT64:
            return HCCL_DATA_TYPE_INT64;
        case DType::MEGRAY_UINT64:
            return HCCL_DATA_TYPE_UINT64;
        case DType::MEGRAY_FLOAT16:
            return HCCL_DATA_TYPE_FP16;
        case DType::MEGRAY_FLOAT32:
            return HCCL_DATA_TYPE_FP32;
        case DType::MEGRAY_FLOAT64:
            return HCCL_DATA_TYPE_FP64;
        case DType::MEGRAY_CHAR:
            return HCCL_DATA_TYPE_INT8;
        default:
            MEGRAY_THROW("invalid dtype");
    }
}

HcclReduceOp as_hccl_reduce_op(ReduceOp rop) {
    switch (rop) {
        case ReduceOp::MEGRAY_SUM:
            return HCCL_REDUCE_SUM;
        case ReduceOp::MEGRAY_MAX:
            return HCCL_REDUCE_MAX;
        case ReduceOp::MEGRAY_MIN:
            return HCCL_REDUCE_MIN;
        default:
            MEGRAY_THROW("invalid reduce mod");
    }
}

}  // namespace MegRay
