#include "megray/common.h"

#include <cstdint>

namespace MegRay {

size_t get_dtype_size(DType dtype) {
    switch (dtype) {
        case MEGRAY_CHAR:
        case MEGRAY_INT8:
        case MEGRAY_UINT8:
            return 1;
        case MEGRAY_FLOAT16:
            return 2;
        case MEGRAY_INT32:
        case MEGRAY_UINT32:
        case MEGRAY_FLOAT32:
            return 4;
        case MEGRAY_INT64:
        case MEGRAY_UINT64:
        case MEGRAY_FLOAT64:
            return 8;
        default:
            MEGRAY_THROW("unknown dtype");
    }
}

}  // namespace MegRay
