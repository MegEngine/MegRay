/**
 * \file src/cncl/utils.cpp
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2023 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "utils.h"
#include "megray/cnrt_context.h"

namespace MegRay {

cnclDataType_t get_cncl_dtype(const DType dtype) {
    switch (dtype) {
        case MEGRAY_INT8:
            return cnclInt8;
        case MEGRAY_UINT8:
            return cnclUint8;
        case MEGRAY_INT32:
            return cnclInt32;
        case MEGRAY_UINT32:
            return cnclUint32;
        case MEGRAY_FLOAT16:
            return cnclFloat16;
        case MEGRAY_FLOAT32:
            return cnclFloat32;
        default:
            MEGRAY_THROW("unknown dtype");
    }
}

cnclReduceOp_t get_cncl_reduce_op(const ReduceOp red_op) {
    switch (red_op) {
        case MEGRAY_SUM:
            return cnclSum;
        case MEGRAY_MAX:
            return cnclMax;
        case MEGRAY_MIN:
            return cnclMin;
        default:
            MEGRAY_THROW("unknown reduce op");
    }
}

}  // namespace MegRay
