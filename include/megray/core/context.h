/**
 * \file src/context.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <memory>
#include <any>

#include "megray/core/common.h"

namespace MegRay {

typedef enum {
    MEGRAY_CTX_DEFAULT = 0,
    MEGRAY_CTX_CUDA = 1,
    MEGRAY_CTX_HIP = 2,
    MEGRAY_CTX_COUNT = 3,
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

} // namespace MegRay
