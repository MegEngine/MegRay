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
