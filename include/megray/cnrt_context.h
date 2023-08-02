/**
 * \file include/megray/cnrt_context.h
 * MegRay is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include <memory>

#include "megray/common.h"
#include "megray/context.h"

#ifdef MEGRAY_WITH_CNCL

#include <cnrt.h>

namespace MegRay {

class CnrtContext : public Context {
public:
    CnrtContext(cnrtQueue_t queue) : m_queue{queue} {}
    static std::shared_ptr<CnrtContext> make(cnrtQueue_t queue) {
        return std::make_shared<CnrtContext>(queue);
    }
    ContextType type() const override { return MEGRAY_CTX_CNRT; }
    cnrtQueue_t get_queue() { return m_queue; }

private:
    cnrtQueue_t m_queue;
};

}  // namespace MegRay

#endif
