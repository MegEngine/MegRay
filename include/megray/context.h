#pragma once

#include "megray/common.h"

namespace MegRay {

typedef enum {
    MEGRAY_CTX_DEFAULT = 0,
    MEGRAY_CTX_CUDA = 1,
    MEGRAY_CTX_HIP = 2,
    MEGRAY_CTX_CNRT = 3,
    MEGRAY_CTX_COUNT = 4,
} ContextType;

/*!
 * MegRay context is an abstraction of communication contexts (e.g. cuda stream)
 * on different platforms, a context should be passed as a parameter when
 * a communicator operation is called
 */
class Context {
public:
    virtual ContextType type() const = 0;
    virtual ~Context() = default;
};

}  // namespace MegRay
